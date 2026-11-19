# -*- coding: utf-8 -*-
"""
PII 脱敏模块（质量底线第三道防线的一部分：防止敏感信息随检索结果外泄）。

设计说明（对齐「零模型、零第三方依赖、硬编码」）：
- 用标准库 re 做正则匹配 + 打码，不引入任何模型与第三方库。
- 覆盖家庭/企业常见敏感字段：手机号、邮箱、身份证、银行卡、密码/token。
- 打码策略：保留首尾少量字符、中间以 * 打码，兼顾可读性与隐私。
- 由 config.SANITIZE_ENABLED 总开关控制是否在 server 输出前启用。
"""

from __future__ import annotations

import re

# 手机号：1[3-9] + 9 位数字，前后不能是数字（避免误伤长数字串）
_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")

# 邮箱：常见 email 形态
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# 身份证：17 位数字 + 1 位数字或 X（18 位）
_ID_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")

# 银行卡：16~19 位连续数字
_BANK_RE = re.compile(r"(?<!\d)\d{16,19}(?!\d)")

# 密码 / 密钥 / token：形如 key=value 或 key: value 的敏感键值对
_SECRET_RE = re.compile(
    r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key)\s*[:=]\s*(\S+)"
)


def _mask_phone(m: re.Match) -> str:
    """手机号打码：保留前 3 后 4，如 138****8000。"""
    s = m.group(0)
    return f"{s[:3]}****{s[-4:]}"


def _mask_email(m: re.Match) -> str:
    """邮箱打码：保留首字符与域名，如 a***@domain.com。"""
    s = m.group(0)
    local, _, domain = s.partition("@")
    head = local[0] if local else ""
    return f"{head}***@{domain}"


def _mask_id(m: re.Match) -> str:
    """身份证打码：保留前 3 后 4，如 110***********1234。"""
    s = m.group(0)
    return f"{s[:3]}{'*' * (len(s) - 7)}{s[-4:]}"


def _mask_bank(m: re.Match) -> str:
    """银行卡打码：保留末 4 位，如 ****1234。"""
    s = m.group(0)
    return f"{'*' * (len(s) - 4)}{s[-4:]}"


def _mask_secret(m: re.Match) -> str:
    """敏感键值对打码：保留键名，值替换为 ***。"""
    return f"{m.group(1)}=***"


def sanitize_text(text: str) -> str:
    """
    对一段文本做 PII 脱敏，返回打码后的文本。

    - 未命中任何敏感模式时原样返回（零开销）。
    - 顺序：身份证/银行卡（长数字）优先于手机号，避免手机号正则先截走长串头部。
    - 各正则相互独立，按顺序替换，互不干扰。
    """
    if not text:
        return text
    text = _ID_RE.sub(_mask_id, text)
    text = _BANK_RE.sub(_mask_bank, text)
    text = _PHONE_RE.sub(_mask_phone, text)
    text = _EMAIL_RE.sub(_mask_email, text)
    text = _SECRET_RE.sub(_mask_secret, text)
    return text
