import re

TERMINAL_TAGS = ("WSHPQFEK",    # Strep-Tag-II
                 "DYKDDDDK",    # FLAG-tag
                 "EQKLISEEDL",  # Myc-tag
                 "MASMTGGQQMG", # T7-tag
                 "KETAAAKFERQHMDS",  # S-tag
                 "GKPIPNPLLGLDST",   # V5-tag
                 "GLNDIFEAQKIEWHE",  # Avi-tag
                )

CLEAVAGE_SITES = (
    ("ENLYFQS",  "Q", "S"),    # TEV        (cut Q|S)
    ("ENLYFQG",  "Q", "G"),    # TEV        (cut Q|G)
    ("LVPRGS",   "R", "G"),    # Thrombin   (cut R|G)
    ("IEGR",     "R", None),   # Factor Xa  (cut after R)
    ("LEVLFQGP", "Q", "G"),    # HRV 3C     (cut Q|G)
    ("DDDDK",    "K", None),   # Enterokinase (cut after K)
)

HIS_MIN = 5           # >= this many consecutive H = His-tag
TERMINAL_WINDOW = 20  # His-run counts as "terminal" if within this many

GS_MIN = 5            # >= this many consecutive G/S = Gly-Ser linker
GS_FLANK_MAX = 2      # linker is "terminal" if <= this many residues to the end


def _cut_offset(motif, p1, p1p):
    """Index of the P1 residue (the cut is *after* this index)."""
    for off, ch in enumerate(motif):
        if ch != p1:
            continue
        if p1p is None:
            if off == len(motif) - 1:
                return off
        elif off + 1 < len(motif) and motif[off + 1] == p1p:
            return off
    return None


def relabel_tags(seq,
                 terminal_tags=TERMINAL_TAGS,
                 cleavage_sites=CLEAVAGE_SITES,
                 his_min=HIS_MIN,
                 window=TERMINAL_WINDOW,
                 gs_min=GS_MIN,
                 gs_flank=GS_FLANK_MAX):
    """Label tag/cleavage-site/linker residues from `seq` alone -- no
    experimental assignment string needed. 'T' = tag/site residues,
    't' = flank toward the terminus, 'P' = proline, 'A' = everything else
    (real protein)."""

    n = len(seq)
    lab = ["P" if aa == "P" else "A" for aa in seq]

    def mark_T(a, b):
        for i in range(max(0, a), min(b, n)):
            lab[i] = "T"

    def mark_t(a, b):
        for i in range(max(0, a), min(b, n)):
            if lab[i] != "T":
                lab[i] = "t"

    # --- His-tag on a terminus: His-run = T, flank toward terminus = t
    for m in re.finditer("H{%d,}" % his_min, seq):
        start, end = m.start(), m.end()
        if start < window:                  # N-terminal
            mark_t(0, start); mark_T(start, end)
        elif end > n - window:              # C-terminal
            mark_T(start, end); mark_t(end, n)

    # --- exact terminal tags (motif = T; flank = t only if terminal)
    for tag in terminal_tags:
        s = 0
        while (i := seq.find(tag, s)) != -1:
            j = i + len(tag)
            if i < window:
                mark_t(0, i); mark_T(i, j)
            elif j > n - window:
                mark_T(i, j); mark_t(j, n)
            else:                            # internal hit: motif only
                mark_T(i, j)
            s = i + 1

    # # --- cleavage sites: cut = tag/protein boundary
    for motif, p1, p1p in cleavage_sites:
        off = _cut_offset(motif, p1, p1p)
        if off is None:
            continue
        recog = motif[:off + 1]              # recognition portion (no scar)
        s = 0
        while (i := seq.find(recog, s)) != -1:
            cut = i + len(recog)
            s = i + 1
            # recog itself drops P1' (e.g. "LVPR" for "LVPR|GS"), so a bare recog
            # match matches natural sequence far more often than the real site --
            # require the actual next residue to be the declared P1' whenever it's
            # present in `seq` (unverifiable only when the sequence ends at `cut`).
            if p1p is not None and cut < n and seq[cut] != p1p:
                continue
            if i < window:                   # N-terminal: tag side is 0..cut
                mark_t(0, i); mark_T(i, cut)
            elif cut > n - window:           # C-terminal: tag side is i..end -- the
                mark_t(cut, n); mark_T(i, cut)  # cut residue is part of the tag side
            else:                            # internal hit: motif only
                mark_T(i, cut)

    # --- Gly/Ser linker just inside a terminal tag -> t
    n_edge = 0
    while n_edge < n and lab[n_edge] in ("t", "T"):
        n_edge += 1
    c_edge = n
    while c_edge > 0 and lab[c_edge - 1] in ("t", "T"):
        c_edge -= 1
    for m in re.finditer("[GS]{%d,}" % gs_min, seq):
        start, end = m.start(), m.end()
        if n_edge <= start <= n_edge + gs_flank:        # abuts N-terminal tag
            mark_t(n_edge, end)
        elif c_edge - gs_flank <= end <= c_edge:        # abuts C-terminal tag
            mark_t(start, c_edge)

    return "".join(lab)