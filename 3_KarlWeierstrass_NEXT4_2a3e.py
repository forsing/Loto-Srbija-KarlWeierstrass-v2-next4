"""
3_KarlWeierstrass_NEXT4_2a3e — POGODNI deo iz 1_KarlWeierstrass_v2.py
Aparat 2a: Brownovo kretanje  +  Test 3e: NIST baterija (NIST-style)

Self-contained:
  - KORAK 1: ucitavanje 4624 izvlacenja i izgradnja f(t) = lex-indeks
  - KORAK 2a: Brown inkrementi (priprema)
  - KORAK 2a3e: NIST-style testovi nad binarizovanim Brown inkrementima
                (monobit, runs, block frequency, cumulative sums,
                 approximate entropy)

Output:
  3_KarlWeierstrass_NEXT4_2a3e.png
  3_KarlWeierstrass_NEXT4_2a3e.txt
"""

import csv
import math
import os
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


T0 = time.time()

CSV_DRAWS = "/Users/4c/Desktop/GHQ/data/loto7_4624_k43.csv"

HERE = os.path.dirname(os.path.abspath(__file__))
PNG_PATH = os.path.join(HERE, "3_KarlWeierstrass_NEXT4_2a3e.png")
TXT_PATH = os.path.join(HERE, "3_KarlWeierstrass_NEXT4_2a3e.txt")

N_MAX = 39
K_PICK = 7
TOTAL_COMBOS = math.comb(N_MAX, K_PICK)


# ─── helperi (samo oni potrebni za 2a + 2a3e) ────────────────────────
def read_loto_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < K_PICK:
                continue
            try:
                nums = tuple(sorted(int(x) for x in row[:K_PICK]))
            except ValueError:
                continue
            if len(nums) == K_PICK and len(set(nums)) == K_PICK:
                rows.append(nums)
    return rows


def lex_rank_1based(combo, n=N_MAX, k=K_PICK):
    """1-based lex indeks (poklapa se sa rednim brojem u kombinacije_39C7.csv)."""
    combo = tuple(sorted(combo))
    rank0 = 0
    prev = 0
    for i, value in enumerate(combo):
        remaining = k - i - 1
        for candidate in range(prev + 1, value):
            rank0 += math.comb(n - candidate, remaining)
        prev = value
    return rank0 + 1


def nist_bits_from_series(series):
    """Binarizacija za NIST-style test: 1 ako je inkrement iznad medijane."""
    x = np.asarray(series, dtype=float)
    med = float(np.median(x))
    bits = (x > med).astype(int)
    if bits.sum() == 0 or bits.sum() == len(bits):
        bits = (x > x.mean()).astype(int)
    return bits


def nist_monobit(bits):
    n = len(bits)
    s_obs = abs(int(np.sum(2 * bits - 1))) / np.sqrt(n)
    p = math.erfc(s_obs / np.sqrt(2))
    return float(p), float(s_obs)


def nist_runs(bits):
    n = len(bits)
    pi = float(bits.mean())
    if abs(pi - 0.5) >= 2 / np.sqrt(n):
        return 0.0, 0, pi
    runs = int(1 + np.sum(bits[1:] != bits[:-1]))
    denom = 2 * np.sqrt(2 * n) * pi * (1 - pi)
    p = math.erfc(abs(runs - 2 * n * pi * (1 - pi)) / denom)
    return float(p), runs, pi


def nist_block_frequency(bits, block_size=128):
    n = len(bits)
    n_blocks = n // block_size
    if n_blocks == 0:
        return float("nan"), float("nan"), 0
    trimmed = bits[:n_blocks * block_size].reshape(n_blocks, block_size)
    props = trimmed.mean(axis=1)
    chi2 = 4 * block_size * float(np.sum((props - 0.5) ** 2))
    p = float(stats.chi2.sf(chi2, n_blocks))
    return p, chi2, n_blocks


def nist_cumulative_sums(bits):
    x = 2 * bits - 1
    walk = np.cumsum(x)
    z = int(np.max(np.abs(walk)))
    if z == 0:
        return 1.0, z, walk
    n = len(bits)
    p = math.erfc(z / np.sqrt(2 * n))
    return float(p), z, walk


def _pattern_counts_circular(bits, m):
    n = len(bits)
    ext = np.concatenate([bits, bits[:m - 1]])
    counts = np.zeros(2 ** m, dtype=float)
    for i in range(n):
        value = 0
        for b in ext[i:i + m]:
            value = (value << 1) | int(b)
        counts[value] += 1
    return counts


def nist_approximate_entropy(bits, m=2):
    n = len(bits)

    def phi(mm):
        counts = _pattern_counts_circular(bits, mm)
        probs = counts[counts > 0] / n
        return float(np.sum(probs * np.log(probs)))

    ap_en = phi(m) - phi(m + 1)
    chi2 = 2 * n * (np.log(2) - ap_en)
    df = 2 ** (m - 1)
    p = float(stats.chi2.sf(chi2, df))
    return p, float(ap_en), float(chi2), df


# ─── KORAK 1: f(t) = lex-indeks ──────────────────────────────────────
draws = read_loto_csv(CSV_DRAWS)
N = len(draws)
lex_idx = np.array([lex_rank_1based(c) for c in draws], dtype=np.float64)

print()
print("3_KarlWeierstrass_NEXT4_2a3e — KORAK 1: formiranje krive f(t)")
print(f"  CSV:                  {CSV_DRAWS}")
print(f"  Ucitano izvlacenja:    {N}")
print(f"  C(39,7):              {TOTAL_COMBOS:,}")
print()

with open(TXT_PATH, "w", encoding="utf-8") as f:
    f.write("3_KarlWeierstrass_NEXT4_2a3e — Brownovo kretanje + NIST baterija (POGODNO)\n")
    f.write("=" * 60 + "\n\n")
    f.write("KORAK 1: Weierstrass-ova funkcija nad svih izvucenih kombinacija\n\n")
    f.write(f"  CSV izvucenih:        {CSV_DRAWS}\n")
    f.write(f"  Ucitano izvlacenja:    {N}\n")
    f.write(f"  C(39,7):              {TOTAL_COMBOS:,}\n")
    f.write("  f(t) = lex-indeks izvucene kombinacije u skupu svih 39C7\n\n")


# ─── KORAK 2a: priprema Brown inkremenata (samo ono sto 2a3e koristi) ─
incr = np.diff(lex_idx)
brown_incr_centered = incr - incr.mean()


# ─── KORAK 2a3e: NIST baterija nad binarizovanim Brown inkrementima ──
T0_2A3E = time.time()

nist_bits = nist_bits_from_series(brown_incr_centered)
nist_n = len(nist_bits)
nist_ones = int(nist_bits.sum())
nist_zeros = int(nist_n - nist_ones)

monobit_p, monobit_s = nist_monobit(nist_bits)
runs_p, runs_count, runs_pi = nist_runs(nist_bits)
block_p, block_chi2, block_count = nist_block_frequency(nist_bits, block_size=128)
cusum_p, cusum_z, cusum_walk = nist_cumulative_sums(nist_bits)
apen_p, apen_value, apen_chi2, apen_df = nist_approximate_entropy(nist_bits, m=2)

nist_rows = [
    ("Monobit frequency", monobit_p, monobit_s),
    ("Runs", runs_p, runs_count),
    ("Block frequency", block_p, block_chi2),
    ("Cumulative sums", cusum_p, cusum_z),
    ("Approx entropy", apen_p, apen_value),
]
nist_pass_count = int(sum(p > 0.05 for _, p, _ in nist_rows if np.isfinite(p)))
nist_total = int(sum(np.isfinite(p) for _, p, _ in nist_rows))

if nist_pass_count == nist_total:
    nist_note = "svi NIST-style testovi prolaze na 0.05"
elif nist_pass_count >= max(1, nist_total - 1):
    nist_note = "uglavnom prolazi, postoji slab signal za proveru"
else:
    nist_note = "vise NIST-style testova pada (moguća struktura / odstupanje)"

print()
print("KORAK 2a3e: Aparat 2a Brownovo kretanje + Test 3e NIST baterija")
print(f"  bits: n={nist_n}  zeros={nist_zeros}  ones={nist_ones}")
for name, p, stat_val in nist_rows:
    print(f"  {name:<20} p={p:.6f}  stat={stat_val}")
print(f"  prolaz: {nist_pass_count}/{nist_total}  ⇒ {nist_note}")
print()

fig2a3e, ax2a3e = plt.subplots(1, 3, figsize=(16, 5))
fig2a3e.suptitle("KORAK 2a3e: Brownovo kretanje + NIST-style testovi  (POGODNO)",
                 fontsize=13, fontweight="bold")

ax2a3e[0].bar(["0", "1"], [nist_zeros, nist_ones], color=["steelblue", "darkorange"])
ax2a3e[0].set_title("Binarizovani Brown inkrementi")
ax2a3e[0].set_xlabel("bit")
ax2a3e[0].set_ylabel("broj")
ax2a3e[0].grid(True, alpha=0.2, axis="y")

names = [row[0] for row in nist_rows]
pvals = np.array([row[1] for row in nist_rows], dtype=float)
colors = ["seagreen" if p > 0.05 else "crimson" for p in pvals]
ax2a3e[1].barh(names, pvals, color=colors)
ax2a3e[1].axvline(0.05, color="black", linestyle="--", linewidth=1.2)
ax2a3e[1].set_xlim(0, 1)
ax2a3e[1].set_title("NIST-style p-vrednosti")
ax2a3e[1].set_xlabel("p-value")
ax2a3e[1].grid(True, alpha=0.2, axis="x")

ax2a3e[2].plot(np.arange(1, nist_n + 1), cusum_walk, linewidth=0.7, color="purple")
ax2a3e[2].axhline(0, color="black", linewidth=0.6)
ax2a3e[2].set_title(f"Cumulative sums walk (z={cusum_z})")
ax2a3e[2].set_xlabel("t")
ax2a3e[2].set_ylabel("cum sum")
ax2a3e[2].grid(True, alpha=0.25)

for a in ax2a3e:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)

fig2a3e.tight_layout()
fig2a3e.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
plt.show()

with open(TXT_PATH, "a", encoding="utf-8") as f:
    f.write("\n")
    f.write("=" * 60 + "\n")
    f.write("KORAK 2a3e: Aparat 2a Brownovo kretanje + Test 3e NIST baterija\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"  PNG:                  {PNG_PATH}\n\n")
    f.write("Binarizacija Brown inkremenata:\n")
    f.write("  bit = 1 ako je centrirani dX iznad medijane, inace 0\n")
    f.write(f"  n bits                = {nist_n}\n")
    f.write(f"  zeros                 = {nist_zeros}\n")
    f.write(f"  ones                  = {nist_ones}\n\n")
    f.write("NIST-style testovi (prolaz ako p > 0.05):\n")
    f.write(f"  {'test':<22}{'p-value':>14}{'stat':>18}{'pass':>10}\n")
    for name, p, stat_val in nist_rows:
        f.write(f"  {name:<22}{p:>14,.8f}{float(stat_val):>18,.8f}{str(p > 0.05):>10}\n")
    f.write("\n")
    f.write("Detalji:\n")
    f.write(f"  Runs pi               = {runs_pi:.8f}\n")
    f.write(f"  Block count           = {block_count}\n")
    f.write(f"  Approx entropy m      = 2\n")
    f.write(f"  Approx entropy chi2   = {apen_chi2:.8f}\n")
    f.write(f"  Approx entropy df     = {apen_df}\n")
    f.write(f"  pass count            = {nist_pass_count}/{nist_total}\n")
    f.write(f"  interpret.            = {nist_note}\n\n")

    elapsed_2a3e = time.time() - T0_2A3E
    f.write(f"Vreme KORAKA 2a3e: {timedelta(seconds=int(elapsed_2a3e))} ({elapsed_2a3e:.1f} s)\n")
    f.write(f"Ukupno vreme:       {timedelta(seconds=int(time.time()-T0))} ({time.time()-T0:.1f} s)\n")

print(f"PNG saved → {PNG_PATH}")
print(f"TXT saved → {TXT_PATH}")
print(f"Vreme KORAKA 2a3e: {timedelta(seconds=int(time.time()-T0_2A3E))} "
      f"({time.time()-T0_2A3E:.1f} s)")
print(f"Ukupno vreme:      {timedelta(seconds=int(time.time()-T0))} "
      f"({time.time()-T0:.1f} s)")
print()
print("KRAJ 3_KarlWeierstrass_NEXT4_2a3e.")
print()
"""
3_KarlWeierstrass_NEXT4_2a3e — KORAK 1: formiranje krive f(t)
  CSV:                  /data/loto7_4624_k43.csv
  Ucitano izvlacenja:   4624
  C(39,7):              15,380,937


KORAK 2a3e: Aparat 2a Brownovo kretanje + Test 3e NIST baterija
  bits: n=4623  zeros=2312  ones=2311
  Monobit frequency    p=0.988266  stat=0.014707472779848216
  Runs                 p=0.000000  stat=3087
  Block frequency      p=0.999997  stat=9.5
  Cumulative sums      p=0.680479  stat=28
  Approx entropy       p=0.000000  stat=0.6297857663526234
  prolaz: 3/5  ⇒ vise NIST-style testova pada (moguća struktura / odstupanje)

PNG saved → /3_KarlWeierstrass_NEXT4_2a3e.png
TXT saved → /3_KarlWeierstrass_NEXT4_2a3e.txt
Vreme KORAKA 2a3e: 0:00:23 (23.1 s)
Ukupno vreme:      0:00:23 (23.1 s)

KRAJ 3_KarlWeierstrass_NEXT4_2a3e.
"""



###############   PREDIKCIJA 4  ###############################

"""
NEXT4 (2a3e, NIST) — predikcija sledećeg bita (gore/dole od medijane) → polu-prostor kandidata.
"""


def lex_unrank_1based(rank, n=N_MAX, k=K_PICK):
    """Vracanje 1-based lex indeksa u Loto 7/39 kombinaciju."""
    rank0 = int(rank) - 1
    combo = []
    prev = 0
    for i in range(k):
        remaining = k - i - 1
        for candidate in range(prev + 1, n + 1):
            count = math.comb(n - candidate, remaining)
            if rank0 >= count:
                rank0 -= count
            else:
                combo.append(candidate)
                prev = candidate
                break
    return tuple(combo)


T0_PRED4 = time.time()

# NIST signal je binarni: bit=1 ako je centrirani dX iznad medijane.
# Runs i ApproxEntropy padaju, zato koristimo kratki bit-kontekst za sledeci bit.
bit_context_len = 2
bits = np.asarray(nist_bits, dtype=int)
current_context = tuple(bits[-bit_context_len:].tolist())

matched_next_indices = []
for i in range(bit_context_len, len(bits)):
    context = tuple(bits[i - bit_context_len:i].tolist())
    if context == current_context:
        matched_next_indices.append(i)

if matched_next_indices:
    matched_next_indices = np.asarray(matched_next_indices, dtype=int)
    pred4_note = "koristi se istorija istog 2-bit konteksta"
else:
    matched_next_indices = np.arange(len(bits), dtype=int)
    pred4_note = "nema poklapanja 2-bit konteksta; koristi se globalna bit distribucija"

matched_bits = bits[matched_next_indices]
p_next_1 = float(np.mean(matched_bits))
pred_bit = int(p_next_1 >= 0.5)

filtered_indices = matched_next_indices[matched_bits == pred_bit]
if len(filtered_indices) == 0:
    filtered_indices = matched_next_indices
    pred4_note += "; nema filtera za pred_bit pa ostaje kontekst distribucija"

next_centered_pool = brown_incr_centered[filtered_indices]
last_lex = float(lex_idx[-1])
last_incr = float(incr[-1])
mean_incr = float(incr.mean())
median_centered = float(np.median(brown_incr_centered))
pred_centered_incr = float(np.mean(next_centered_pool))
pred_incr = mean_incr + pred_centered_incr
pred_lex_float = last_lex + pred_incr
pred_lex = int(np.clip(round(pred_lex_float), 1, TOTAL_COMBOS))
pred_combo = lex_unrank_1based(pred_lex)

quantile_grid = [0.10, 0.25, 0.50, 0.75, 0.90]
candidate_rows = []
seen_lex = set()
for q in quantile_grid:
    cand_centered = float(np.quantile(next_centered_pool, q))
    cand_incr = mean_incr + cand_centered
    cand_lex = int(np.clip(round(last_lex + cand_incr), 1, TOTAL_COMBOS))
    if cand_lex in seen_lex:
        continue
    seen_lex.add(cand_lex)
    candidate_rows.append((q, cand_incr, cand_lex, lex_unrank_1based(cand_lex)))

print()
print("PREDIKCIJA 4 — NEXT4 / 2a3e / NIST / sledeci bit inkrementa")
print(f"  bit kontekst           = {current_context}")
print(f"  istorijskih prelaza    = {len(matched_next_indices)}")
print(f"  P(next bit=1)          = {p_next_1:.6f}")
print(f"  pred. bit              = {pred_bit}")
print(f"  median centered dX     = {median_centered:,.2f}")
print(f"  zadnji lex             = {int(last_lex):,}")
print(f"  zadnji inkrement       = {last_incr:,.2f}")
print(f"  pred. inkrement        = {pred_incr:,.2f}")
print(f"  pred. lex              = {pred_lex:,}")
print(f"  pred. kombinacija      = {pred_combo}")
print(f"  napomena               = {pred4_note}")
print("  kvantil kandidati iz pred. bit polu-prostora:")
for q, cand_incr, cand_lex, combo in candidate_rows:
    print(f"    q={q:>4.2f}  dX={cand_incr:>14,.2f}  lex={cand_lex:>10,}  combo={combo}")
print()

with open(TXT_PATH, "a", encoding="utf-8") as f:
    f.write("\n")
    f.write("=" * 60 + "\n")
    f.write("PREDIKCIJA 4: NEXT4 / 2a3e / NIST / sledeci bit inkrementa\n")
    f.write("=" * 60 + "\n\n")
    f.write("Model:\n")
    f.write("  NIST runs i approximate entropy padaju, pa bitovi nisu potpuno random.\n")
    f.write("  Bit = 1 ako je centrirani dX iznad medijane.\n")
    f.write("  Poslednja 2 bita daju kontekst za procenu sledeceg bita.\n")
    f.write("  Zatim se prognoza bira iz istorijskih dX vrednosti tog bit polu-prostora.\n\n")
    f.write("Parametri:\n")
    f.write(f"  bit kontekst           = {current_context}\n")
    f.write(f"  broj prelaza           = {len(matched_next_indices)}\n")
    f.write(f"  P(next bit=1)          = {p_next_1:.8f}\n")
    f.write(f"  pred. bit              = {pred_bit}\n")
    f.write(f"  bit pool n             = {len(next_centered_pool)}\n")
    f.write(f"  median centered dX     = {median_centered:,.8f}\n")
    f.write(f"  mean(dX)               = {mean_incr:,.8f}\n")
    f.write(f"  zadnji lex             = {int(last_lex):,}\n")
    f.write(f"  zadnji inkrement       = {last_incr:,.8f}\n")
    f.write(f"  pred. centrirani dX    = {pred_centered_incr:,.8f}\n")
    f.write(f"  pred. inkrement        = {pred_incr:,.8f}\n")
    f.write(f"  napomena               = {pred4_note}\n\n")
    f.write("Glavna prognoza:\n")
    f.write(f"  pred. lex float        = {pred_lex_float:,.8f}\n")
    f.write(f"  pred. lex              = {pred_lex:,}\n")
    f.write(f"  pred. kombinacija      = {pred_combo}\n\n")
    f.write("Kvantil kandidati iz pred. bit polu-prostora:\n")
    f.write(f"  {'q':>8}{'dX':>18}{'lex':>14}  kombinacija\n")
    for q, cand_incr, cand_lex, combo in candidate_rows:
        f.write(f"  {q:>8.2f}{cand_incr:>18,.8f}{cand_lex:>14,}  {combo}\n")
    f.write("\n")
    elapsed_pred4 = time.time() - T0_PRED4
    f.write(f"Vreme PREDIKCIJE 4: {timedelta(seconds=int(elapsed_pred4))} ({elapsed_pred4:.1f} s)\n")

print(f"TXT updated → {TXT_PATH}")
print(f"Vreme PREDIKCIJE 4: {timedelta(seconds=int(time.time()-T0_PRED4))} "
      f"({time.time()-T0_PRED4:.1f} s)")
print()

"""
Predikcija iz NIST/binary signala: prvo procena sledećeg bita inkrementa, zatim kandidati iz odgovarajuće istorijske distribucije.

koristi NIST bitove iz Brown inkremenata
poslednja 2 bita su kontekst
procenjuje P(next bit=1)
bira sledeći bit i uzima istorijske inkremente iz tog bit polu-prostora
računa glavnu Loto kombinaciju + kvantil-kandidate
upisuje u 3_KarlWeierstrass_NEXT4_2a3e.txt
"""


"""
3_KarlWeierstrass_NEXT4_2a3e — KORAK 1: formiranje krive f(t)
  CSV:                  /data/loto7_4624_k43.csv
  Ucitano izvlacenja:   4624
  C(39,7):              15,380,937


KORAK 2a3e: Aparat 2a Brownovo kretanje + Test 3e NIST baterija
  bits: n=4623  zeros=2312  ones=2311
  Monobit frequency    p=0.988266  stat=0.014707472779848216
  Runs                 p=0.000000  stat=3087
  Block frequency      p=0.999997  stat=9.5
  Cumulative sums      p=0.680479  stat=28
  Approx entropy       p=0.000000  stat=0.6297857663526234
  prolaz: 3/5  ⇒ vise NIST-style testova pada (moguća struktura / odstupanje)

PNG saved → /3_KarlWeierstrass_NEXT4_2a3e.png
TXT saved → /3_KarlWeierstrass_NEXT4_2a3e.txt
Vreme KORAKA 2a3e: 0:00:03 (3.6 s)
Ukupno vreme:      0:00:03 (3.6 s)

KRAJ 3_KarlWeierstrass_NEXT4_2a3e.


PREDIKCIJA 4 — NEXT4 / 2a3e / NIST / sledeci bit inkrementa
  bit kontekst           = (0, 0)
  istorijskih prelaza    = 767
  P(next bit=1)          = 0.745763
  pred. bit              = 1
  median centered dX     = -2,463.74
  zadnji lex             = 513,114
  zadnji inkrement       = -2,143,496.00
  pred. inkrement        = 6,483,105.12
  pred. lex              = 6,996,219
  pred. kombinacija      = (3, x, 21, y, 26, z, 33)
  napomena               = koristi se istorija istog 2-bit konteksta
  kvantil kandidati iz pred. bit polu-prostora:
    q=0.10  dX=  1,088,757.10  lex= 1,601,871  combo=(1, x, 15, y, 18, z, 37)
    q=0.25  dX=  3,141,040.25  lex= 3,654,154  combo=(2, x, 12, y, 21, z, 38)
    q=0.50  dX=  6,549,929.00  lex= 7,063,043  combo=(4, x, 6, y, 24, z, 31)
    q=0.75  dX=  9,583,324.25  lex=10,096,438  combo=(6, x, 11, y, 23, z, 26)
    q=0.90  dX= 11,449,206.30  lex=11,962,320  combo=(7, x, 20, y, 24, z, 32)

TXT updated → /3_KarlWeierstrass_NEXT4_2a3e.txt
Vreme PREDIKCIJE 4: 0:00:00 (0.0 s)
"""
