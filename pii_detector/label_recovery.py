"""Recover misspelled / morphological PII label variants back onto the prompt's
own vocabulary.

WHY: the LLM is shown a fixed category list (the keys of ``PII_DEFINITIONS`` in
the ``templates`` module, plus any categories the caller injects via
``extra_definitions``) but frequently emits near-misses — ``DATE`` for
``DATES``, ``PRIVATE_NUMBER`` for ``PERSONAL_NUMBERS``, ``EMAIL`` for
``CONTACT_INFO``. Those land in the "unsupported labels" tally and, worse,
canonicalise to ``UNK`` instead of the bucket the model clearly meant —
depressing type-aware precision/recall for no real reason.

HOW: a single ``recover_label(raw, valid_labels)`` resolves one emitted label
against the set the model was actually shown, in two deterministic, ordered
tiers (the second only consulted if the first misses):

  1. morphology — generated variants of the emission: the exact form (after
     upper-casing + whitespace strip), its singular<->plural toggle on a
     trailing 's' (``DATE`` <-> ``DATES``), and the taxonomy's PRIVATE<->PERSONAL
     swap (``PRIVATE_NUMBER`` -> ``PERSONAL_NUMBERS``), in every combination.
  2. curated semantic aliases — hand-verified equivalences that morphology
     can't reach (``EMAIL`` -> ``CONTACT_INFO``, ``SSN`` -> ``IDENTIFICATION``).

NO fuzzy / edit-distance matching: the recovery must be auditable and never
silently re-home a genuinely novel label. Anything neither tier can resolve is
returned as ``UNRECOVERED_LABEL`` — so a true hallucination is marked
unrecoverable rather than being force-fitted (the caller preserves the original
emitted string separately so it stays visible).

The valid set is passed in (derived from the live prompt's rendered category
block) rather than hardcoded here, so recovery tracks whatever vocabulary the
model was shown — the base categories or any caller-supplied
``extra_definitions`` — without this module knowing the taxonomy.
"""

# Sentinel label assigned to an emission that none of the recovery tiers could
# map back to the prompt vocabulary. The ORIGINAL emitted label is preserved
# separately on the prediction (see PiiDetector.__call__) so the report's
# Unsupported-Labels tally still shows the real hallucinated string — UNK is
# only the canonicalisation-facing label.
UNRECOVERED_LABEL = "UNK"


# Curated semantic aliases, keyed by the EXPECTED prompt label -> the set of
# emitted strings that mean it. ONLY genuinely non-derivable jumps live here:
# anything reachable by pluralisation or the PRIVATE<->PERSONAL swap is
# generated automatically (see `_morph_variants`) and must NOT be listed.
# Variants are written as the model emits them (UPPER/underscored); their
# plurals are covered by the morphology tier and need no separate entry.
#
# Every expected-label KEY is a category that templates.py actually defines
# (PII_DEFINITIONS) — NOT a coarse canonical bucket. A variant absent from the
# shown prompt's `valid_labels` is skipped at lookup time, so a target the model
# wasn't given is never produced; if none of a variant's targets are valid it
# falls to UNK.
#
# A single variant may appear under MORE THAN ONE expected label: `recover_label`
# scans the groups in INSERTION ORDER and takes the first expected label that
# both lists the variant and is present in `valid_labels` — so insertion order
# defines candidate priority (e.g. a bare ADDRESS would lean to whichever
# address bucket is listed first).
_ALIAS_GROUPS = {
    # email / phone -> the prompt's combined contact bucket
    # (CONTACT_INFO == "personal email addresses, personal phone numbers").
    "CONTACT_INFO": {
        "EMAIL", "EMAIL_ADDRESS", "PERSONAL_EMAIL",
        "PHONE", "PHONE_NUMBER", "PHONENUMBER",
    },
    # account/customer identifiers. (PRIVATE_NUMBER/PERSONAL_NUMBER and plurals
    # are reached by `_morph_variants`; only non-derivable synonyms here.)
    "PERSONAL_NUMBERS": {
        "ACCOUNT_NUMBER", "CUSTOMER_ID", "ACCOUNT_ID",
        "USER_ID", "ORDER_ID",
        "TRANSACTION_ID",
        "ITEM_ID",
        "CONSIGNMENT_ID", "PRODUCT_ID", "PLAN_ID", "COURIER_ID",
    },
    # names of private individuals (the prompt distinguishes PRIVATE vs PUBLIC
    # names; a bare "personal name" leans private). PERSONAL_NAME/PRIVATE_NAME
    # and plurals come from the swap generator; only synonyms here.
    "PRIVATE_NAMES": {"NAME", "PERSON", "PERSON_NAME", "FIRST_NAME", "LAST_NAME"},
    # date-like -> DATES ("birth dates, appointment dates, anniversary dates").
    "DATES": {"DOB", "DATE_OF_BIRTH", "DAYS", "DATE_TIME"},
    # identification document numbers ("SSNs, driver's license, passport,
    # employee/student IDs").
    "IDENTIFICATION": {
        "SSN", "SOCIAL_SECURITY_NUMBER", "PASSPORT",
        "LICENSE_NUMBER", "EMPLOYEE_ID",
    },
    # credentials -> CREDENTIALS. (CREDENTIAL->CREDENTIALS is plural-derivable;
    # only non-derivable synonyms here.)
    "CREDENTIALS": {"PASSWORD", "API_KEY", "ACCESS_KEY"},
    # Addresses. A bare/misspelled ADDRESS is offered to BOTH split halves
    # (FULL first); PARTIAL_ADDRESS / FULL_ADDRESS go only to their matching
    # half. (PARTIAL_ADDRESS needs "+ES", out of the single-'s' morphology
    # tier's reach, so it lives here rather than being generated.)
    "FULL_ADDRESSES": {"ADDRESS", "ADRESS", "ADRESSES", "ADDRESS_ID"},
    # A generic public web page -> PUBLIC_URL (the model's own PRIVATE_URL /
    # PUBLIC_URL are already valid and pass the exact-match tier untouched).
    "PUBLIC_URL": {"WEB_PAGE"},
    # payment methods -> FINANCIAL.
    "FINANCIAL": {"PAYMENT_METHOD", "PAYMENT_METHOD_ID"},
    # device/account identifiers not reachable by morphology.
    "DIGITAL_IDS": {"USERNAME"},
    # geographic region references -> LOCATION.
    "LOCATION": {"REGION"},
}


def _norm(label: str) -> str:
    """Canonical comparison form: upper-cased, outer whitespace stripped.

    Matches how `valid_labels` is normalised in `recover_label`, so membership
    tests compare like with like regardless of the model's casing/padding."""
    return label.upper().strip()


def _morph_variants(norm: str):
    """Yield morphological variants of ``norm`` to test against ``valid_labels``,
    in priority order (closest first).

    Two derivations, in every combination:
      * pluralise / de-pluralise on a trailing 's' (DATE<->DATES),
      * the PRIVATE<->PERSONAL split the taxonomy uses (the prompt names the
        identifier bucket PERSONAL_NUMBERS but private individuals PRIVATE_NAMES,
        so a model that writes PRIVATE_NUMBER or PERSONAL_NAME is one swap away).

    Generated rather than enumerated so new variants in that family (e.g. a
    future PRIVATE_<X> emission) recover for free without touching the table.
    ``norm`` itself is yielded first, so exact matches are covered here too.
    """
    bases = [norm]
    if "PRIVATE" in norm:
        bases.append(norm.replace("PRIVATE", "PERSONAL"))
    if "PERSONAL" in norm:
        bases.append(norm.replace("PERSONAL", "PRIVATE"))
    seen: set = set()
    for base in bases:
        # base, then its plural toggle (de-pluralise guarded on len>1 so a bare
        # "S" doesn't collapse to "").
        toggled = base[:-1] if (base.endswith("S") and not base.endswith("SS") and len(base) > 1) else (base + "S" if not base.endswith("SS") else base + "ES")
        for cand in (base, toggled):
            if cand and cand not in seen:
                seen.add(cand)
                yield cand


def recover_label(raw: str | None, valid_labels: frozenset[str]) -> str:
    """Resolve one emitted label to the prompt vocabulary, or ``UNRECOVERED_LABEL``.

    ``valid_labels`` is the set the model was shown (already the canonical
    UPPER/stripped form — see ``valid_label_set``). Returns the recovered valid
    label, or ``UNRECOVERED_LABEL`` when nothing matches (so the caller never
    has to translate a ``None`` into the sentinel itself).

    Tiers, first hit wins:
      1. morphology — exact match, plural<->singular, PRIVATE<->PERSONAL
         (generated by ``_morph_variants``),
      2. curated semantic alias (whose target must itself be valid).
    """
    if raw is None:
        return UNRECOVERED_LABEL
    norm = _norm(raw)
    if not norm:
        return UNRECOVERED_LABEL

    # Pre-compute morphological variants once; used by both tiers.
    morph = list(_morph_variants(norm))

    # Tier 1: generated morphological variants (includes the exact form).
    for cand in morph:
        if cand in valid_labels:
            return cand

    # Tier 2: curated alias. Scan the groups in insertion order (which defines
    # priority when a variant belongs to several) and return the first expected
    # label that both lists this variant (or a morph variant of it) AND is in
    # the shown vocabulary — so a split-vocab or Ext-only target stays unrecovered
    # under a prompt lacking it.
    for expected, variants in _ALIAS_GROUPS.items():
        if any(m in variants for m in morph) and expected in valid_labels:
            return expected

    return UNRECOVERED_LABEL


def valid_label_set(rendered_categories: str) -> frozenset[str]:
    """Extract the valid label set from a rendered PII-category block.

    The block is what ``templates.build_pii_categories`` produces — one line per
    category, formatted as ``" - LABEL: definition"`` (or ``" - LABEL"`` when a
    category has no definition). We pull the token between the leading ``-`` and
    the first ``:`` on each such line. Deriving the set from the actual rendered
    prompt (rather than importing PII_DEFINITIONS) keeps recovery correct for
    any caller-injected ``extra_definitions`` — the target is always exactly
    what the model was shown.

    Returns labels in the normalised (UPPER/stripped) form `recover_label`
    compares against.
    """
    labels = set()
    for line in rendered_categories.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        body = stripped[2:]
        label = body.split(":", 1)[0].strip()
        if label:
            labels.add(_norm(label))
    return frozenset(labels)
