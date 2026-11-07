"""Hand-written red-flag rules for recruitment scams common in India / South Asia.

The training data (EMSCAD, mostly US/UK postings from 2012-14) almost never contains these
patterns, so the text model cannot learn them. Any rule hit forces a "review" alert
regardless of the model score.

Measured hit rates when written (Nov 2026):
  EMSCAD (17,880 real postings):  22/17,014 legitimate (0.13%), 2/866 fraud (0.2%)
  Aioshi/smolified-fakejob (597 synthetic Indian postings): 147/373 fakes (39%), 0/224 reals
The India numbers are optimistic: these rules were written after reading that dataset.
"""
import re

RULES = {
    # Candidate must pay something to get the job.
    'upfront_fee': (
        r'\b(registration|security|processing|training|joining|refundable|interview|kit|uniform'
        r'|verification|onboarding|certification)\s+(fees?|deposits?|charges?)\b'
        r'|\b(deposit|transfer|send)\s+(of\s+)?(rs\.?|inr|₹|rupees)\s*\d'
        r'|\d[\d,]*\s*(rs|inr|rupees|/-)\b[^.]{0,40}\b(fee|deposit|charges?|registration)'
        r'|\b(must|need to|have to|required to)\s+(pay|deposit|buy|purchase)\b'
        r'|\bpay\b[^.]{0,40}\bbefore\s+(the\s+)?(interview|joining|selection|training|start)'),
    # Hiring moved to a private chat app instead of an official channel.
    'chat_app_contact': (
        r'\b(whats\s?app|telegram)\s*(us|me|now|at|on|number|no\.?|:|\+|\d)'
        r'|\b(contact|message|text|ping|reach|call|dm|send|apply|connect|join)\b[^.]{0,30}'
        r'\b(on|via|through|over|to|in)\s+(our\s+)?(whats\s?app|telegram)'
        r'|\bwa\.me\b|\bt\.me/'),
    # Asks for identity documents or banking secrets.
    'id_or_bank_request': (
        r'aadh?aa?r|\bpan\s+(card|number|details)|bank\s+(account\s+)?details'
        r'|\b(share|send)\s+(your\s+)?(otp|cvv|pin)\b'),
    # Promises a job without a real selection process.
    'guaranteed_job': (
        r'guarantee[ds]?\s+(a\s+)?(job|offer|placement|selection|income)'
        r'|\bno\s+interview|without\s+(an?\s+)?interview|100%\s+(job|placement)'),
}
_COMPILED = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in RULES.items()}


def scam_flags(text):
    """Names of the rules that fire on this text (empty list if none)."""
    return [name for name, pattern in _COMPILED.items() if pattern.search(text or '')]


def any_flag(texts):
    """1 where at least one rule fires, else 0, for an iterable of texts."""
    return [int(bool(scam_flags(t))) for t in texts]
