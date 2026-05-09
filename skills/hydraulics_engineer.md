# Hydraulic Valve Domain Knowledge

You are a hydraulic systems engineer specialising in directional control valves.
Use this knowledge when extracting product data from manufacturer user guides,
matching competitor products to Danfoss equivalents, and answering technical queries.

This document is the canonical reference for the agent. It is loaded by section —
each `## N. <name>` block is retrieved by keyword (e.g. agents asking for "spool"
will receive every section whose name contains "spool"). Each section is therefore
self-contained.

---

## 1. Ordering Code Structure

Every directional valve has an **ordering code** — a segmented model number where
each position encodes one configurable option. The ordering code is the single
source of truth for what a product is.

### Reading an ordering code table

The user guide shows a **template** at the top — the full code with position markers:

```
Bosch Rexroth 4WE6 example:
4   WE  6  E   6X  /   E   G24   N9    K4
01  02  03 04  05  06  07  08    09    10
```

Below the template, each **position number** has a sub-table listing every allowed
option. Each row in a position's sub-table is one **option within that segment**,
NOT a separate product.

### The critical rule: segments vs products

- Total products = `options_in_seg_1 × options_in_seg_2 × … × options_in_seg_N`
- Each row in a segment table is ONE OPTION for that segment
- DO NOT treat each row as a complete product
- DO NOT collapse all options into a single product
- The ordering code template tells you the segment ORDER — preserve it exactly

### "no code" / blank entries

Many segments have a default option marked "no code" or left blank. This means:
- The segment position is OMITTED from the final ordering code when the default is selected
- It is still a valid option — include it with value `""` or `"standard"`
- Example: "Without manual override = no code" means the override segment is simply absent
- Example: Bosch Rexroth E1 in 4WE6 RE 23178 — ordering code uses spool letter directly without separator

### Segment positions are manufacturer-specific

- Bosch Rexroth 4WE6 (RE 23178): position 04 = symbol/spool letter, position 08 = voltage
- Danfoss DG4V-3: different positions
- NEVER assume a fixed position — always read the template header first

### Worked example: Danfoss `DG4V-3-2C-M-U-H7-60`

| Pos | Code | Meaning |
|---|---|---|
| 1 | DG4V | Wet-armature solenoid directional valve (D=directional, G=gasket-mounted, 4V=4-way variable spool) |
| 2 | 3 | Size 6 / NG06 / D03 / CETOP 3 mounting |
| 3 | 2C | Spool variant — closed centre, no crossover in transition (BLOCKED canonical pattern) |
| 4 | M | Special-features identifier (modifications follow) |
| 5 | U | DIN 43650 Form-A connector (3-pin industrial) |
| 6 | H7 | 24V DC coil (H family, 7 = 60 Hz wiring) |
| 7 | 60 | Design series (interchangeable with 61, 62) |

For other DG4V suffixes you may encounter:
- `FS` — Soft Shift (factory orifice, 200–500 ms shift). **Do not substitute with standard valve.**
- `P` — Proportional control (requires amplifier)
- `MU` — Manual override included
- `X90` — Explosion-proof (ATEX/IECEx)
- `PVG` — Integrated pressure-reducing section

### Worked example: Bosch Rexroth `4WE6E62/EG24N9K4`

| Pos | Code | Meaning |
|---|---|---|
| 1 | 4 | 4 main ports |
| 2 | WE | Directional control, wet-armature, solenoid-actuated |
| 3 | 6 | Size NG06 (CETOP 3) |
| 4 | E | Spool symbol — closed centre (BLOCKED canonical pattern) → equivalent to Danfoss `2C` |
| 5 | 62 | Design generation (6X series, generation 2) |
| 6 | / | Separator |
| 7 | E | Style (electrical version — central plug E) |
| 8 | G24 | 24V DC solenoid (G family for DC) |
| 9 | N9 | Manual override option |
| 10 | K4 | DIN 43650 connector type |

### Manufacturer layout differences

- **Bosch Rexroth** (e.g. 4WE6, RE 23178): text-rich, highly structured tables.
  Ordering code uses numbered positions (01–21) with sub-tables. Spool symbols on a
  dedicated page with ISO diagrams. Electrical options split across 4 pages: DC
  individual, DC central, AC individual, AC central. Voltage/connector matrix uses
  checkmarks. "no code" used extensively for default options.
- **Danfoss / Vickers** (e.g. DG4V-3): graphics-heavy. Most content embedded in
  images, not text. Ordering code breakdown shown as a visual diagram. Spool diagrams
  on dedicated pages with minimal text labels. Some pages contain ONLY images with
  no extractable text — vision is required.
- **General approach**: read the ENTIRE guide first to build a mental map of the
  document structure. Find the ordering code template FIRST — this is the skeleton.
  For each segment, find its option table (may be on a different page). Find the
  spool/symbol page and map each diagram to its code. Extract technical data
  (pressure, flow, weight) from the specs table. Record electrical options from
  the voltage/connector matrix.

---

## 2. Spool Type Identification — Canonical Functional Taxonomy

The spool (control spool / symbol) defines the valve's flow paths. This is the
PRIMARY differentiator between valve variants and the most important field for
cross-manufacturer matching.

### Match by hydraulic function, not by code letter

Each manufacturer uses different codes for the same physical spool behaviour. The
agent must reduce every spool to a **canonical pattern** in the form
`FAMILY|sol-a-energised|sol-b-energised`. Matching uses string equality on the
canonical pattern; if patterns equal exactly, spool match = 1.0.

### The seven canonical pattern families

| Family | De-energised centre | Typical use |
|---|---|---|
| **BLOCKED** | All four ports blocked (P, A, B, T) | Load-holding closed-centre circuits |
| **OPEN** | All four ports interconnected (P↔A↔B↔T) | Pump unloaded, work ports vented |
| **TANDEM** | P→T connected; A and B blocked | Pump unloaded, work ports held |
| **FLOAT** | P blocked; A→T, B→T (work ports drain to tank) | Free-floating cylinders, semi-open |
| **REGEN** | P→A and P→B linked; T blocked | Regenerative cylinder extension |
| **SELECTOR** | P→A (or P→B); other work port blocked; T isolated | 3-way selection in a 4-way body |
| **ASYMMETRIC** | Centre is asymmetric (one work port behaves differently from the other), or energised positions are not mirror images | Specialised cylinder controls (e.g. regen-on-extend only) |

### Reading ISO 1219-1 spool symbols

User guides show spool symbols as **rectangular boxes** side by side. **The number
of boxes does not always equal the number of valve positions** — see "topology"
below.

Inside each box:
- **Lines connecting ports** = flow path open
- **T-bar or dead-end at a port** = port blocked
- **Arrows** = direction of permitted flow
- Ports are labelled: **P** (pressure), **T** (tank/return), **A** and **B** (work ports)

To classify a spool's family from the diagram:
1. Identify the topology (4/3 vs 4/2 — see below).
2. Look at the **de-energised centre / spring-offset position** (the box that
   represents the spool with no solenoid energised).
3. List which ports are connected and which are blocked.
4. Match against the seven canonical families above.
5. Record both the manufacturer's own code (e.g. `E`, `2C`, `22A`) AND the
   canonical pattern.

### Topology — 4/2 single-solenoid vs 4/3 double-solenoid

A 3-box symbol can mean **either** of two things:
- **4/3 double-solenoid valve**: three discrete spool positions, both solenoids
  exist (left = sol-A energised, centre = both de-energised / spring-centred, right
  = sol-B energised).
- **4/3 single-solenoid spring-centred valve**: three discrete positions but only
  one solenoid; the unreachable side is shown for completeness.
- **4/2 single-solenoid drawn with a 3-box symbol** for documentation reasons:
  the centre/transition state is illustrated even though only two stable
  positions exist. Bosch Rexroth does this for spools A, C, D.

A 2-box symbol almost always means **4/2 single-solenoid** with spring offset
(Bosch Rexroth B, Y).

The agent must record:
- `topology`: `"4/2 single-solenoid"`, `"4/3 double-solenoid"`, or `"4/2 double-solenoid"`
- `hand_build`: `"left"`, `"right"`, or `"symmetric"` — for asymmetric or single-solenoid spools.
  4/3 double-solenoid valves are `symmetric` by default.

### Hand build — terminal "L" suffix in Danfoss codes

In Danfoss DG4V naming, a terminal `L` suffix on a spool code means
**spring-offset left-hand build**. Examples:
- `2A` = closed centre, RH build → `2AL` = same family, **LH build**
- `22A` = selector, RH build → `22AL` = selector mirror, LH build

LH and RH builds are **functionally mirrors** — the centre and energised positions
swap. They are NOT interchangeable because the manifold port orientation determines
which build fits. The matching agent should treat differing `hand_build` as a
significant mismatch even when the canonical pattern is identical.

### Spool code appears in the ordering code

The spool type occupies one segment position in the ordering code. When extracting:
- Record the manufacturer's own code (e.g. `E`, `2C`, `22A`)
- Record the canonical pattern (e.g. `BLOCKED|PA-BT|PB-AT`)
- For Danfoss, record any terminal `L` separately as `hand_build = "left"`

---

## 3. Bosch Rexroth 4WE6 Spool Reference (Domain-Verified Table)

This table covers all 20 standard spool letters in Bosch Rexroth RE 23178
(WE6 directional valve, 2019-01 edition). It is the reference of record for
Rexroth-side identification.

| Code | Topology | De-energised centre / position | Canonical pattern |
|---|---|---|---|
| **A** | 4/2 single-solenoid (RH build) | P→A; B blocked; T isolated. Energised: P→B, A blocked, T isolated. T port not used (3-way function in 4-way body) | `SELECTOR\|PA-BB-TI\|PB-AB-TI` |
| **B** | 4/2 single-solenoid (LH build) | Mirror of A — opposite-hand build for installation flexibility | `SELECTOR\|PA-BB-TI\|PB-AB-TI` |
| **C** | 4/2 single-solenoid (RH) | P→A, B→T (alt transition vs D) | `OPEN\|PA-BT\|PB-AT` |
| **D** | 4/2 single-solenoid (RH) | P→A, B→T centre; sol-A energised: P→B, A→T (standard 4/2 mirror) | `BLOCKED\|PA-BT\|PB-AT` |
| **E** | 4/3 double-solenoid | All four ports blocked | `BLOCKED\|PA-BT\|PB-AT` |
| **E1** | 4/3 double-solenoid | All blocked WITH P→A/B pre-opening in transition. ⚠ Pressure intensification risk with differential cylinders. | `BLOCKED\|PA-BT\|PB-AT` |
| **F** | 4/3 double-solenoid | P→A, P→T linked; B blocked. Hybrid open/tandem. **No clean Danfoss equivalent.** | `HYBRID-OPEN-TANDEM\|PA-BT\|PB-AT` |
| **G** | 4/3 double-solenoid | P→T; A & B blocked (pump unloaded, work ports held) | `TANDEM\|PA-BT\|PB-AT` |
| **H** | 4/3 double-solenoid | All four ports interconnected (P, A, B, T all connected) | `OPEN\|PA-BT\|PB-AT` |
| **J** | 4/3 double-solenoid | P blocked; A→T, B→T (work ports drain) | `FLOAT\|PA-BT\|PB-AT` |
| **L** | 4/3 double-solenoid | P blocked; B blocked; A→T (only A drains). **No clean Danfoss equivalent.** | `ASYMMETRIC-DRAIN-A\|PA-BT\|PB-AT` |
| **M** | 4/3 double-solenoid | P→A, P→B; T blocked (regenerative) | `REGEN\|PA-BT\|PB-AT` |
| **P** | 4/3 double-solenoid | {P, B, T} linked; A blocked (pump unloaded via B port; A held). **No clean Danfoss equivalent.** | `ASYMMETRIC-TANDEM-B\|PA-BT\|PB-AT` |
| **Q** | 4/3 double-solenoid | P blocked; A→T, B→T with metering on transition. **Functionally identical to W.** | `FLOAT\|PA-BT\|PB-AT` |
| **R** | 4/3 double-solenoid | All blocked centre; sol-B energised asymmetric: P→A linked B, T blocked (regen on extend) | `ASYMMETRIC\|PA-BT\|PA-LB-TB` |
| **T** | 4/3 double-solenoid | P→T; A & B blocked (alt transition vs G). **No exact Danfoss equivalent in user guide.** | `TANDEM-ALT-TRANSITION\|PA-BT\|PB-AT` |
| **U** | 4/3 double-solenoid | P blocked; A blocked; B→T (only B drains) | `ASYMMETRIC\|PA-BT\|PB-AT` |
| **V** | 4/3 double-solenoid | All ports open with metering on A and B. **No clean Danfoss equivalent.** | `OPEN-METERED\|PA-BT\|PB-AT` |
| **W** | 4/3 double-solenoid | P blocked; A→T, B→T metered. **Functionally identical to Q.** | `FLOAT\|PA-BT\|PB-AT` |
| **Y** | 4/2 single-solenoid (LH) | P→B, A→T (mirror of D) | `BLOCKED\|PA-BT\|PB-AT` |

### Rexroth suffix conventions

- **`73` suffix** (e.g. `E73`, `G73`, `H73`, `J73`) = **soft-shift / smooth-switching**
  variant. Minimises hydraulic hammer / shock during shifting at the cost of slower
  shift time (200–500 ms vs 25–40 ms standard). The `73` variant has the **same**
  canonical pattern as its base spool but should NOT be auto-substituted with a
  non-soft-shift Danfoss equivalent — drop confidence by ~10% and force human
  review for shock-sensitive applications.
- **`E1`, `E2`, `E3`** = transition-behaviour variants of E (E1 has P→A/B pre-opening
  per RE 23178 footnote 2, with explicit pressure-intensification warning).
- **`46` suffix** (e.g. `C46`, `D46`) = special-version variants. Per RE 23178
  footnote 3: only with version SO407 and OF.
- **Letter + position**: ordering codes append a position letter (`A` for spool
  position "a", `B` for position "b") to the spool letter, e.g. `..EA..` or
  `..E73A..`. The position letter is part of the full ordering code, not part of
  the spool functional identity.

### Rexroth-internal duplicates

These exist as separate codes but are functionally equivalent:
- **G ≡ T** — same de-energised centre (P→T, A&B blocked); differ only in transition
  behaviour. Both TANDEM family.
- **C ≡ D** — same de-energised centre (P→A, B→T) and same energised position; differ
  only in transition behaviour.
- **Q ≡ W** — no functional difference based on domain-expert review; treat as aliases.

---

## 4. Spool Cross-References — Rexroth ↔ Danfoss with False Friends

The cross-reference below is **domain-expert ground truth**. The matching agent
uses these to translate from a Rexroth code to its Danfoss equivalent. Matches are
performed via canonical-pattern equality at runtime through `spool_seed.json`.

### Confidence policy when no canonical match

When a Rexroth spool's canonical pattern does not equal any Danfoss canonical
pattern (e.g. unique patterns like `HYBRID-OPEN-TANDEM`, `ASYMMETRIC-DRAIN-A`,
`ASYMMETRIC-TANDEM-B`, `TANDEM-ALT-TRANSITION`, `OPEN-METERED`):
- spool_function_match = 0.0
- Final confidence drops below the 0.75 threshold → "contact sales rep" path
- This is intentional — F, L, P, T, V have no equivalent in the Danfoss user guide

### Rexroth → Danfoss authoritative mapping

| Rexroth | Danfoss | Confidence | Notes |
|---|---|---|---|
| A | `22A` | high | Selector spool, RH build |
| B | `22AL` | high | Selector spool, LH build (mirror of A) |
| C | `0A` | high | 4/2 single-solenoid, alt transition vs D |
| D | `2A` | high | 4/2 single-solenoid, RH build |
| E | `2C` | high | Closed centre |
| E1 | `2C` or `2N` | medium | E with P-A/B pre-opening — flag intensification risk |
| F | — | none | No DG4V equivalent → contact sales rep |
| G | `8C` | high | Tandem — common substitution |
| H | `0C` | high | **⚠ FALSE-FRIEND** — Rexroth H ≠ Danfoss H |
| J | `6C` | high | Float (semi-open) — work ports drain |
| L | — | none | No DG4V equivalent |
| M | `7C` (also `4C`) | high | Regenerative — both REGEN family |
| P | — | none | No DG4V equivalent |
| Q | `6C` (with metered transition) | medium | Aliased with W |
| R | `52C` | high | Asymmetric regen-on-extend |
| T | — | none | TANDEM family, but no exact Danfoss equivalent |
| U | `31C` | medium | Similar function with different transitions |
| V | — | none | No DG4V equivalent |
| W | `33C` | high | Float metered — aliased with Q |
| Y | `2AL` | high | 4/2 single-solenoid, LH build |

### False friends (DO NOT confuse)

These are codes that look equivalent across manufacturers but ARE NOT.

- **Rexroth `H` ≠ Danfoss `H`.** Rexroth H = **all four ports interconnected**
  (true OPEN). Danfoss H = "Float center: A, B and T connected, P blocked"
  (FLOAT family). They share the letter but mean different families. Rexroth H
  maps to Danfoss `0C`, not Danfoss `H`.
- **Older catalogue `8C` → "semi-open"** is wrong against the current Danfoss
  user guide. Danfoss `8C` is **TANDEM** (P→T, A&B blocked) — the mapping target
  for Rexroth G.
- **Older catalogue `6C` → "tandem"** is wrong against the current Danfoss user
  guide. Danfoss `6C` is **FLOAT (semi-open)** — the mapping target for Rexroth J.
- **Older catalogue `33C` → "partial crossover"** is wrong. Danfoss `33C` is
  **FLOAT with metered transition** — the mapping target for Rexroth W.
- **Old `2AL` description as "lapped spool"** is wrong. `2AL` is a **4/2
  single-solenoid spring-offset LH build** — the mapping target for Rexroth Y.

### Soft-shift substitution warning

Replacing a soft-shift valve (Rexroth 73 variants, Danfoss `FS` suffix) with a
standard valve causes:
- Pressure spikes 80–150 bar above operating pressure
- Acceleration spikes 15–25 m/s² (vs 2–4 m/s² with soft-shift)
- Hose life reduction 60–70% from fatigue cycling
- +15–20 dB(A) noise increase

**Always match shift-dynamics across manufacturers** — do not auto-substitute
Rexroth E73 with Danfoss `2C` standard. If a soft-shift Rexroth comes in,
either find a Danfoss soft-shift counterpart or drop confidence below 0.75.

### Coil mechanical incompatibility

Even when valve bodies are dimensionally interchangeable (DG4V-3 ↔ 4WE6 ↔ D1VW
share ISO 4401-03 mounting), **coils are NOT cross-compatible** between
manufacturers:
- Vickers/Eaton coil: 42 × 40 × 35 mm rectangular housing
- Bosch Rexroth coil: Ø30 × 55 mm cylindrical housing

Forcing a coil swap damages connector pins and can crack the coil housing. The
agent should never recommend a cross-manufacturer coil substitution.

### 4/2 single-solenoid selector concept (Rexroth A/B → Danfoss 22A/22AL)

These spools use a 4-way valve body but the T port is **isolated in both spool
positions**. They function as 3-way selector valves: pump pressure routes to
either A or B; the unselected work port is blocked; tank port is unused. They
are NOT interchangeable with standard 4-way DG4V spools.

---

## 5. Key Specification Fields

When extracting from a user guide, capture these fields (in priority order):

| Field | Where to find it | Unit | Example |
|---|---|---|---|
| category | Title / first page | — | `directional_valve` |
| valve_type | Title / features section | — | 4/3, 4/2, 3/2 |
| valve_size | Ordering code segment | ISO 4401 size | 6 (NG6/D03/CETOP 3) |
| max_pressure_bar | Technical data table | bar | 350 |
| max_flow_lpm | Technical data table | l/min | 80 |
| spool_type | Symbol page + ordering code | manufacturer code | `E`, `2C`, `22A` |
| spool_function | Derived from spool diagram | canonical family | BLOCKED, OPEN, TANDEM, FLOAT, REGEN, SELECTOR, ASYMMETRIC |
| topology | Symbol diagram | — | `4/3 double-solenoid`, `4/2 single-solenoid` |
| hand_build | Symbol + spool code suffix | — | `left`, `right`, `symmetric` |
| coil_voltage | Electrical tables in ordering code | V | 24 (DC), 230 (AC 50/60Hz) |
| voltage_type | Electrical tables | — | DC, AC |
| electrical_connection | Ordering code segment | manufacturer code | `K4`, `DL`, `U` (DIN 43650 Form A) |
| manual_override | Ordering code segment | manufacturer code | `N`, `N9`, `MU` |
| seal_material | Ordering code segment | — | NBR, FKM, HNBR |
| mounting_pattern | Features / technical data | ISO standard | ISO 4401-03 |
| weight_kg | Technical data | kg | 1.95 |

### DG4V-3 vs DG4V-5 spec reference

| Spec | DG4V-3 (Size 6 / D03) | DG4V-5 (Size 10 / D05) |
|---|---|---|
| ISO mounting | 4401-03-02-0-05 (CETOP 3) | 4401-05-04-0-05 (CETOP 5) |
| Bolt pattern | 31.75 × 31.75 mm (M6 × 1.0) | 50 × 50 mm (M8 × 1.25) |
| Port size | G 1/4″ BSP or 7/16-20 UNF | G 3/8″ BSP or 9/16-18 UNF |
| Max P (P, A, B) | 350 bar | 315 bar |
| Max P (T return) | 210 bar | 160 bar |
| Peak P (transient ≤2 s) | 420 bar | 380 bar |
| Max Q standard | 40–60 l/min | 80–100 l/min |
| Max Q "H" series | 80 l/min | 120 l/min |
| Response time | 25–40 ms | 30–50 ms |
| Coil power | 19–22 W | 28–32 W |
| Weight | 1.8–2.2 kg | 3.2–3.8 kg |

### Dimensional interchange (ISO 4401)

| ISO 4401 | NG | CETOP | Dxx | Compatible series |
|---|---|---|---|---|
| 4401-03 | NG6 | CETOP 3 | D03 | Danfoss DG4V-3 ↔ Rexroth 4WE6 ↔ Parker D1VW |
| 4401-05 | NG10 | CETOP 5 | D05 | Danfoss DG4V-5 ↔ Rexroth 4WE10 ↔ Parker D3W/D3VW |
| 4401-07 | NG16 | CETOP 7 | D07 | (larger valves) |
| 4401-08 | NG25 | CETOP 8 | D08 | (industrial-press scale) |

Bodies bolt to the same subplate — but coils, internal cartridges, and surface
finish are NOT cross-compatible. Use this table to compute size matches; never
to recommend coil replacement.

### Where specs live in the guide

- **Page 1**: product name, category, key features, headline pressure/flow
- **Ordering code pages** (typically pages 2–6): every configurable option with its code
- **Symbols page**: spool diagrams with codes — vision is required for graphics-only pages
- **Technical data pages**: operating limits, fluid compatibility, weights
- **Electrical pages**: voltage/connector compatibility matrix (often a checkmark grid)
- **Performance / characteristic curves**: flow vs pressure drop per spool
- **Dimensions pages**: physical dimensions and mounting pattern
- Multilingual layout cues for "ordering code" sections: "How to Order",
  "Model Code", "Type Designation", "Bestellangaben", "Bestellschlüssel",
  "Codice di ordinazione".

### Suffix convention summary

| Suffix | Meaning | Notes |
|---|---|---|
| `FS` (Danfoss/Vickers) | Soft-shift, factory orifice | Do not substitute with standard |
| `73` (Rexroth) | Soft-shift / smooth-switching | Same — drop confidence on cross-substitution |
| `P` | Proportional control | Requires amplifier |
| `MU` | Manual override included | |
| `X90` | Explosion-proof (ATEX/IECEx) | Hazardous-area certification |
| `PVG` | Integrated pressure-reducing section | |
| `C` (Vickers position 4) | Spring-centred return | Returns to neutral when de-energised |
| `A` (Vickers position 4) | Spring-offset to one position | |
| `D` (Vickers position 4) | Detent (holds last energised position) | |
| terminal `L` (Danfoss) | Spring-offset, **left-hand build** | NOT lapped spool |

---

## 6. Unit Normalisation

Always normalise to these canonical forms:

| Raw value | Normalised |
|---|---|
| `24VDC`, `24 V DC`, `G24`, `H7` | `coil_voltage`: 24, `voltage_type`: DC |
| `12VDC`, `G12`, `G7` | `coil_voltage`: 12, `voltage_type`: DC |
| `W230`, `230V 50/60Hz`, `B` | `coil_voltage`: 230, `voltage_type`: AC |
| `W110`, `110V 60Hz`, `A` | `coil_voltage`: 110, `voltage_type`: AC |
| `350 bar`, `5076 psi` | `max_pressure_bar`: 350 |
| `80 l/min`, `21 US gpm` | `max_flow_lpm`: 80 |
| `1.95 kg`, `4.3 lbs` | `weight_kg`: 1.95 |
| `G3/8`, `3/8" BSP` | `port_size`: G3/8 |
| `9/16-18 UNF`, `7/16-20 UNF` | `port_size`: SAE (record both BSP and SAE alternates) |
| `ISO 4401-03-02-0-05` | `mounting_pattern`: ISO 4401-03 |
| `NG6`, `D03`, `Size 6`, `CETOP 3` | `valve_size`: 6 |

### Voltage code hierarchy

Danfoss/Vickers coil codes follow a two-tier convention:

**Tier 1 — voltage family (single character):**
- `G` = 12V DC family
- `H` = 24V DC family
- `A` = 110V AC family
- `B` = 220V AC family (some sources show `B` for 110V — context-dependent;
  consult the specific Coil Rating section of the user guide being parsed)

**Tier 2 — specific variant (family + frequency code):**
- `G7` = 12V DC, 60 Hz wiring (1.6 A steady state)
- `H7` = 24V DC, 60 Hz wiring (0.8 A steady state)
- (other digits indicate other frequency/connector combinations)

The agent should accept and parse both tiers. When a code appears as a single
letter (`G`, `H`, `A`, `B`) without a digit, infer from the Coil Rating section of
the user guide being parsed.

**Diagnostic resistance fingerprints** (useful for the chat agent answering
troubleshooting questions, not for matching):
- 24V DC coil ≈ 26 Ω
- 12V DC coil ≈ 6.5 Ω
- 110V AC coil ≈ 95 Ω

### Bosch Rexroth voltage codes

- `G12` = 12V DC, `G24` = 24V DC, `G48` = 48V DC
- `W110` = 110V AC, `W230` = 230V AC (50/60 Hz where indicated)

### Numeric parsing rules

`_parse_numeric_if_possible()` only converts values that ARE pure numbers (or
number+unit like `315 bar`). Preserves mixed alpha-numeric values: `24VDC`,
`G3/8`, `FKM`, `ISO 4401-03` are returned as strings, not corrupted.

### Letter / digit ambiguity (OCR and vision sources)

OCR and vision models confuse `0/O`, `1/I/l`, `5/S`, `2/Z`, `8/B`. Rule of thumb
for ordering codes: **digits dominate** size, voltage, flow, design-number
segments; **letters dominate** spool, seal, override, connector segments. Use
the segment's role from the ordering-code template to disambiguate.

---

## 7. Common Failure Modes and False Friends

### What NOT to do during extraction

1. **DO NOT** treat each row in an ordering code sub-table as a separate complete
   product. Each row is one option for one segment. Products are combinations
   across segments.
2. **DO NOT** assume segment positions are the same across manufacturers. Always
   read the ordering code template to learn the segment order for THIS guide.
3. **DO NOT** match spool types by code letter alone. Rexroth `E` and Danfoss
   `2C` are the same spool (closed centre) but different codes. Always match by
   canonical pattern.
4. **DO NOT** confuse Rexroth `H` with Danfoss `H` — they are the most common
   false-friend in this domain. Rexroth H = OPEN family → maps to Danfoss `0C`.
5. **DO NOT** ignore "no code" / blank / standard entries. These are valid
   default options and must be included in the product combinations.
6. **DO NOT** invent specifications not present in the guide. If a value is not
   stated, leave the field null — do not guess.
7. **DO NOT** confuse operating-pressure limits with test pressure. Use the
   "maximum operating pressure" value, not burst/proof/test pressure.
8. **DO NOT** mix DC and AC electrical options. They have different connector
   types, voltages, and switching characteristics. Keep them as separate
   product variants.
9. **DO NOT** skip the spool-diagram page. The spool type cannot be reliably
   extracted from text alone — the ISO symbol diagrams are the authoritative
   source for flow-path behaviour.
10. **DO NOT** treat a 3-box symbol as automatically 4/3 double-solenoid.
    Check whether one or both solenoid sides have a coil drawn — if only one,
    the valve is single-solenoid (Bosch Rexroth A, C, D are 4/2 with 3-box
    diagrams).

### What NOT to do during matching

11. **DO NOT** auto-substitute soft-shift (`73` / `FS`) variants with standard
    valves. The shift dynamics are different and pressure spikes can damage
    the system.
12. **DO NOT** ignore `topology` when matching. A 4/2 single-solenoid spool
    is not interchangeable with a 4/3 double-solenoid even when their canonical
    patterns happen to be equal.
13. **DO NOT** ignore `hand_build`. LH and RH builds are mirrors and require
    the manifold port orientation to match. A `2AL` is not a drop-in for a `2A`.
14. **DO NOT** recommend cross-manufacturer coil substitution. Coils have
    different dimensions and connector pin layouts even when valve bodies are
    interchangeable per ISO 4401.
15. **DO NOT** report a high-confidence match when the canonical pattern is
    unique to one manufacturer (e.g. Rexroth F/L/P/T/V have unique patterns
    that intentionally don't map to any Danfoss spool — these should drop
    confidence below the 0.75 threshold).
16. **DO NOT** extract data from cross-reference / competitor-equivalence
    tables as if they were the manufacturer's own ordering codes. These tables
    advertise compatibility with other brands and are NOT the canonical product
    definition.

### Common AC coil failure mode (for chat agent)

AC coils draw 4–5× steady-state current at inrush. If the spool sticks (typically
from contamination), current stays at inrush level, generating ~137 W of heat
versus ~22 W normal operation. Class F insulation (155 °C) fails when temperature
exceeds 180 °C, and the coil burns out open-circuit. DC coils have built-in
current limiting and do not burn out under similar stuck-spool conditions. Always
recommend **24V DC where reliability matters** for distributor enquiries about
coil failures.
