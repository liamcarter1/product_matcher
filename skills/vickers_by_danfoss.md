# Vickers by Danfoss DG4V-3 Domain Knowledge

Manufacturer-specific extraction guidance for Vickers by Danfoss DG4V-3 solenoid
operated directional control valves.  Load this file in addition to
`hydraulics_engineer.md` whenever the manufacturer is identified as "Vickers",
"Vickers by Danfoss", "Eaton/Vickers", or the series is "DG4V-3".
The universal spool taxonomy, canonical patterns, and Rexroth↔Danfoss cross-
reference tables live in `hydraulics_engineer.md` — this file covers DG4V-3-
specific ordering code structure, datasheet layout, and extraction failure modes.

---

## 1. Product Identity

| Field | Value |
|---|---|
| Brand | **Vickers by Danfoss** (formerly Eaton/Vickers; rebranded 2021) |
| Series | DG4V-3 |
| Type | Solenoid operated directional valve |
| Standard | ISO 4401 size 03 / CETOP 3 / NFPA D-03 / ANSI B93.7M-D03 |
| Document ref | BC442280316180en-000201 (Sep 2024) |
| Compatible bodies | Mounts on same subplate as Rexroth 4WE6, Parker D1VW (ISO 4401-03) |

---

## 2. DG4V-3 Ordering Code — Complete Structure (15 Positions)

Page 3 of the user guide (BC442280316180en-000201) shows a model code description
table with 16 numbered rows.  Section 16 (Special features: EN21, EN38 approval codes)
is excluded here as it is not relevant to standard product matching.  The 15 working
sections assemble as follows, with hyphens as separators between the major groups:

### Template (from page 3 of BC442280316180en-000201)

```
** DG4V—3  (*)  -  **  *(L)  -  (**)  -  (V)  M  -  (S*)  -  ****  D*  (L)  -  *  *  -  6*
 1    2      3      4    5        6        7    8     9        10    11   12   13  14    15
```

Boxes 1–16 are numbered in the guide; section 16 `(EN***)` is excluded here (special
approval codes not relevant to standard matching).

Separators: each `-` in the template is a literal hyphen in the assembled code.
Positions shown in `( )` are optional — when blank they are **omitted from the code**
along with their adjacent separator if the entire group becomes empty.

### All 15 positions

| Pos | Segment name | is_fixed | Valid codes (from page 3 guide table) |
|---|---|---|---|
| 01 | `seal_type` | false | blank = Viton (omitted); `F6` = Buna Nitrile / High CAN |
| 02 | `series` | **true** | `DG4V-3` — D=Directional, G=Gasket mounted, 4=Solenoid operated, V=350 bar rated (P/A/B), 3=ISO4401 Size D03 |
| 03 | `performance` | false | blank = default performance (omitted); `R` = Standard performance with 8-watt coil; `H` = High performance; `S` = Standard performance, X5 only |
| 04 | `spool_type` | false | Numeric identifier from Page 4 spool symbols: `0`, `2`, `3`, `6`, `8`, `12`, `22`, `23`, `35`, `52`… |
| 05 | `spool_spring` | false | `A` = spring offset end-to-end (RH); `AL` = same, LH build; `B` = spring offset end-to-centre (RH); `BL` = same, LH build; `C` = spring centred; `N` = no spring / detented |
| 06 | `manual_override` | false | blank = plain override at solenoid end(s) (omitted); `H` = waterresistant override; `Z` = no overrides at either end; `W` = twist and lock override; `A` = no override at non-solenoid end (single solenoid only) |
| 07 | `vm_modifier` | false | blank = standard M orientation (omitted); `V` = VM style — solenoid A at port B end. **Mandatory for spool type 8.** See page 4 for solenoid identification table. |
| 08 | `flag_symbol` | **true** | `M` (always — indicates electrical options follow) |
| 09 | `spool_switch` | false | blank = none (omitted); `S7` = spool position monitoring switch, single solenoid valves only (see coil types 6 and 10); `S9` = switch per separate catalogue AF458770480968en-000101 |
| 10 | `coil_type` | false | `U` = ISO4400 DIN43650; `U1` = ISO4400 with PG11 plug; `KU` = top exit flying lead 150mm; `FW` = flying lead ½" NPT; `FTW` = flying lead terminal block ½" NPT; `FPA3W` = 3-pin connector ½" NPT; `FPA5` = 5-pin connector ½" NPT; `KUP4` = Junior Timer AMP; `KUP5`/`KUP6` = Packard/Deutsch; `X5` = explosion proof |
| 11 | `indicator_light` | false | blank = none (omitted); `L` = solenoid indicator lights (flying lead coil types only, excluding FPA\*\*W) |
| 12 | `surge_suppressor` | false | blank = none (omitted); `D1` = diode positive bias; `D2` = diode negative bias; `D7` = Transorb type — see Page 14 for circuit details |
| 13 | `coil_rating` | false | **Single or two-letter code — no tank pressure digit.** `G` = 12VDC; `GL` = 12VDC variant; `H` = 24VDC; `HL` = 24VDC variant; `HM` = 24VDC 8-watt (standard performance DG4V-3-R); `DS` = 28VDC 30-watt; `B` = 110VAC 50Hz/120VAC 60Hz; `D` = 220VAC 50Hz/240VAC 60Hz |
| **14** | **`tank_pressure_code`** | **false** | **Single digit — NOT a bar value, but a T-port pressure specification code.** See table below. 7 is the most common (DC high performance at 207 bar / 3000 psi). |
| 15 | `design_number` | false | `60` = Basic design; `61` = Type 8 spool (mandatory for spool types starting with 8) |

### Section 14 — Tank pressure rating codes (from page 3 guide table)

The four codes at section 14 specify the T-port maximum pressure rating.  The digit
is a **code**, not a pressure value — code 7 ≠ 7 bar:

| Code 14 | T-port max pressure | Applies to | Restriction |
|---|---|---|---|
| `4` | 70 bar (1000 psi) | All models | **X5 coil type only** ▲ |
| `6` | 207 bar (3000 psi) | AC high performance models, incl. S7 switch | — |
| `7` | 207 bar (3000 psi) | **DC high performance models, incl. S7 switch** | — **(most common)** |
| `8` | 160 bar (2300 psi) | AC high performance models, lower T-port rating | — |

Guide note: "Refer to Operating Data for port T pressure ratings."

Code 7 is the most common because the majority of DG4V-3 valves are DC-powered
high-performance models.  In the assembled code `H7`, `G7`, `B6`, etc.:
- The letter (pos 13) is the coil rating
- The digit (pos 14) is the tank pressure code
They are **concatenated with no separator** between them.

### Worked examples — correct position mapping

**Standard DC valve:** `DG4V-3-2C-M-U-H7-60`

| Code | Pos | Meaning |
|---|---|---|
| `DG4V-3` | 02 | Series (fixed) |
| `2` | 04 | Spool type 2 |
| `C` | 05 | Spring centred |
| `M` | 08 | Flag symbol (fixed) |
| `U` | 10 | DIN 43650 connector |
| `H` | 13 | Coil: 24VDC |
| `7` | 14 | Tank pressure code: DC high performance, 207 bar |
| `60` | 15 | Basic design |

Positions 01, 03, 06, 07, 09, 11, 12 all blank → omitted. Result: `DG4V-3-2C-M-U-H7-60` ✓

---

**Spool 8, VM style:** `DG4V-3-8C-VM-U-H7-61`

| Code | Pos | Meaning |
|---|---|---|
| `8` | 04 | Spool type 8 |
| `C` | 05 | Spring centred |
| `V` | 07 | VM modifier — **mandatory for spool 8** |
| `M` | 08 | Flag symbol (fixed) |
| `U` | 10 | DIN connector |
| `H` | 13 | 24VDC |
| `7` | 14 | DC high performance, 207 bar |
| `61` | 15 | Type 8 spool design — **mandatory for spool 8** |

Result: `DG4V-3-8C-VM-U-H7-61` ✓

---

**LH build:** `DG4V-3-2AL-M-U-H7-60`

| Code | Pos | Meaning |
|---|---|---|
| `2` | 04 | Spool type 2 |
| `A` + `L` | 05 | Spring offset end-to-end, **left-hand build** |
| `H` | 13 | 24VDC |
| `7` | 14 | DC HP, 207 bar |
| `60` | 15 | Basic design |

Positions 04+05 concatenate to `2AL`. Result: `DG4V-3-2AL-M-U-H7-60` ✓

---

**Standard 8-watt coil:** `DG4V-3-R2C-M-U-HM7-60`

| Code | Pos | Meaning |
|---|---|---|
| `R` | 03 | Standard performance (8-watt coil) |
| `2` | 04 | Spool type 2 |
| `C` | 05 | Spring centred |
| `HM` | 13 | Coil: 24VDC 8-watt (standard performance) |
| `7` | 14 | DC HP, 207 bar |
| `60` | 15 | Basic design |

Result: `DG4V-3-R2C-M-U-HM7-60` ✓

### Spool 8-series mandatory constraints

Both constraints MUST appear in the `"constraints"` array whenever spool type 8 is present:

```json
[
  {
    "when_segment": "spool_type",
    "when_value_regex": "^8",
    "enforce_segment": "vm_modifier",
    "enforce_value": "V"
  },
  {
    "when_segment": "spool_type",
    "when_value_regex": "^8",
    "enforce_segment": "design_number",
    "enforce_value": "61"
  }
]
```

Page 4 of the guide: "*Spool 8 only offered in VM style nomenclature."
Generating `DG4V-3-8C-M-U-H7-60` is an invalid code Danfoss cannot supply.

---

## 3. Spool Types Reference Table

The functional symbols page (page 4 of the user guide) shows spool types with ISO 1219
hydraulic symbols.  All spools are available in **M-style**; only spool 8* is **VM-style only**.

**NOTE from guide**: "Bolded spool numbers have standard lead time" — these are
the highest-volume, shortest-lead-time variants and should be prioritised in generation.

### Standard lead time spools (listed bold in guide)

| Spool code | Topology | De-energised centre | Canonical pattern | Notes |
|---|---|---|---|---|
| **2C** | 4/3 double-solenoid | All ports blocked | `BLOCKED\|PA-BT\|PB-AT` | Most common; closed centre |
| **2A** | 4/2 single-solenoid (RH) | P→A, B→T | `BLOCKED\|PA-BT\|PB-AT` | Spring-offset RH build |
| **2AL** | 4/2 single-solenoid (LH) | Mirror of 2A | `BLOCKED\|PA-BT\|PB-AT` | Spring-offset LH build |
| **0A** | 4/3 double-solenoid | All ports interconnected | `OPEN\|PA-BT\|PB-AT` | Open centre |
| **0B** | 4/2 single-solenoid | P→A, B→T (spring return) | `OPEN\|PA-BT\|PB-AT` | Open centre spring offset |
| **6B** | 4/2 single-solenoid | P→T; A & B blocked | `TANDEM\|PA-BT\|PB-AT` | Tandem; single solenoid |
| **3B** | 4/2 single-solenoid | P blocked; A→T, B→T | `FLOAT\|PA-BT\|PB-AT` | Float / semi-open |
| **22A** | 4/2 single-solenoid (RH) | P→A; B blocked; T isolated | `SELECTOR\|PA-BB-TI\|PB-AB-TI` | Selector / 3-way function |
| **22AL** | 4/2 single-solenoid (LH) | Mirror of 22A | `SELECTOR\|PA-BB-TI\|PB-AB-TI` | Selector LH build |
| **8C** ⚠ | 4/3 double-solenoid | All ports blocked (special transition) | `BLOCKED\|PA-BT\|PB-AT` | **VM only; design 61 mandatory** |

### Extended spool list (in appendix or performance data pages)

| Spool code | Topology | Centre condition | Canonical pattern |
|---|---|---|---|
| **2B** | 4/2 single-solenoid (RH) | Spring offset to B end | `BLOCKED\|PA-BT\|PB-AT` |
| **2BL** | 4/2 single-solenoid (LH) | Spring offset to B end, LH | `BLOCKED\|PA-BT\|PB-AT` |
| **2N** | 4/3 detented | All ports blocked, detented | `BLOCKED\|PA-BT\|PB-AT` |
| **6C** | 4/3 double-solenoid | P blocked; A→T, B→T | `FLOAT\|PA-BT\|PB-AT` | 
| **8AL** | 4/2 single-solenoid (RH) | All blocked, special transition | `BLOCKED\|PA-BT\|PB-AT` | VM only; design 61 |
| **23A** | 4/2 single-solenoid | Selector variant | `SELECTOR\|PA-BB-TI\|PB-AB-TI` | |
| **35A** | 4/2 single-solenoid | Selector variant | `SELECTOR\|PA-BB-TI\|PB-AB-TI` | |
| **52B** | 4/2 single-solenoid | Asymmetric (regen on extend) | `ASYMMETRIC\|PA-BT\|PA-LB-TB` | |

### Spool suffix conventions (Danfoss)

| Suffix | Meaning | Example |
|---|---|---|
| `C` | Spring centred (returns to neutral centre) | `2C`, `0C`, `8C` |
| `A` | Spring offset, end-to-end (right-hand build) | `2A`, `22A`, `8AL` |
| `AL` | Spring offset, end-to-end, **left-hand** build | `2AL`, `22AL` |
| `B` | Spring offset, end-to-centre | `2B`, `6B`, `3B` |
| `BL` | Spring offset, end-to-centre, LH build | `2BL` |
| `N` | No spring, detented | `2N` |

**Important**: `L` suffix = spring-offset **left-hand build**.  It does NOT mean "lapped"
spool.  LH and RH builds are NOT interchangeable — the manifold port orientation
determines which fits.

### M vs VM orientation

Page 4 of the guide defines the solenoid orientation for each style:

| Style | Solenoid A location | Energise Sol. A → | Energise Sol. B → |
|---|---|---|---|
| **M** | On the B-port side (left end) | P flows to A-port | P flows to B-port |
| **VM** | On the A-port side (right end) | A-port drains to T | B-port drains to T |

This matters for circuit design (VM is preferred for differential cylinders where draining
the rod side extends the cylinder under gravity or pilot pressure).  For pure cross-
reference matching, M and VM with the same spool code are the same canonical pattern —
they share port paths but differ in which solenoid triggers which direction.

---

## 4. Connector / Coil Types

### From page 12 (Electrical Plugs and Connectors)

| Code | Type | Description | IP | Notes |
|---|---|---|---|---|
| **U** / **U1** / **U6** | DIN 43650 connector | Standard screw-terminal DIN plug; cable Ø6–10mm, wire 0.5–1.5mm² | IP65 | Most common; can be rotated 90° on valve |
| **KU** | Top exit flying lead | DIN connector body with leads exiting upward | IP65 | Space-saving where side exit not possible |
| **KUP4** | Junior Timer AMP | AMP Mate-N-Lok sealed connector | IP67 | Automotive-grade sealed |
| **KUP5** / **KUP8** | Deutch / Packard | Deutch connector (KUP5) or Packard with seals (KUP8) | IP67 | Higher vibration/moisture rating |
| **X5** | Explosion proof | Sealed ATEX/IECEx coil; no external connector | EX | Only available with specific voltages; excludes FPA**W seals |
| **M12** | M12 4-pin connector | M12 × 1 round connector; pin 4 = +V, pin 3 = 0V; pins 1 & 2 unused for coil | IP65 | Used for spool position indicator (S7) connection |
| **F1** / **F1T** | Flying lead / terminal | 2-wire flying lead (F1) or with terminal block (F1T); approx 300mm long | IP54 | UL/CSA listed option |
| **FT1** / **FT1W** | Flying lead with terminal strip | Terminal strip with multiple wires | IP54 | "W" = with grounding |
| **KUP7** | Packard connector (male pins) | Male Packard pins; 2-wire, 24VDC or 12VDC coils | — | |

### DIN 43650 connector detail

- Cable diameter range: Ø6–10 mm (0.24–0.40 in)
- Wire section range: Ø0.5–1.5 mm² (0.0008–0.0023 in²)
- Terminals: screw type
- Type of protection: IEC 144 class IP65 when plugs are correctly fitted with interface seals
- Connector can be rotated 90° at intervals on valve

### NFPA PA configuration (from page 13)

Some installations use the PA configuration — the terminal plug is oriented with leads
exiting to the left (A-port side).  This is an installation preference, not a separate
ordering code.

---

## 5. Coil Rating Codes (Position 13)

Position 13 of the ordering code is the **coil rating** — a one or two letter code.
It does **not** include a digit.  The digit that follows it in the assembled code
(position 14) is the tank pressure code, not part of the coil rating.

### Coil rating codes (from page 3, section 13)

| Code | Voltage | Type | Watt | Notes |
|---|---|---|---|---|
| **G** | 12V | DC | — | Standard 12VDC coil |
| **GL** | 12V | DC | — | 12VDC variant |
| **H** | 24V | DC | — | **Most common DC coil** |
| **HL** | 24V | DC | — | 24VDC variant |
| **HM** | 24V | DC | 8W | Standard performance (DG4V-3-R); associated with R at position 3 |
| **DS** | 28V | DC | 30W | High-wattage DC |
| **B** | 110V | AC | — | 50Hz/120VAC 60Hz |
| **D** | 220V | AC | — | 50Hz/240VAC 60Hz |

### Common combinations seen in practice

Because the most common tank pressure code is `7` (DC high performance, 207 bar),
the codes `G7`, `H7`, `B6`, `D6` appear frequently in assembled ordering codes.
These are NOT two-character voltage codes — they are pos 13 + pos 14 concatenated:

| Assembled | Pos 13 (coil) | Pos 14 (tank code) | Meaning |
|---|---|---|---|
| `H7` | H = 24VDC | 7 = DC HP 207 bar | 24VDC, DC high performance |
| `G7` | G = 12VDC | 7 = DC HP 207 bar | 12VDC, DC high performance |
| `B6` | B = 110VAC | 6 = AC HP 207 bar | 110VAC, AC high performance |
| `D6` | D = 220VAC | 6 = AC HP 207 bar | 220VAC, AC high performance |
| `HM7` | HM = 24VDC 8W | 7 = DC HP 207 bar | 24VDC 8-watt standard performance |

### AC coil characteristics (from page 5, Operating Data)

| Spec | At 50 Hz | At 60 Hz |
|---|---|---|
| Dual frequency power | 200 W | 250 W |

AC coils draw 4–5× steady-state current at inrush.  Recommend DC coils for
reliability-critical applications.

---

## 6. Operating Specifications

From the Operating Data table on page 5 of BC442280316180en-000201:

| Specification | Value |
|---|---|
| Max pressure — P, A, B ports | **315 bar** (4571 psi) |
| Max pressure — T port (standard) | **70 bar** (1000 psi) |
| T-port back-pressure rating options | **4, 6, 7, 8 bar** (ordered separately) |
| Max flow — DC models | Up to **80 l/min** (21 USgpm) at rated conditions |
| Max flow — AC 50Hz | Approx. 40 l/min |
| Operating temperature | −20°C to +80°C (−4°F to +176°F); special option to +100°C |
| Fluid type | Mineral oil (ISO 11158 or equivalent); synthetic OK if NBR-compatible |
| Viscosity range | 7–450 cSt (46–2090 SUS) operating; 15–36 cSt optimal |
| Seals (standard) | Viton (FKM/FPM) — order code `blank` |
| Seals (optional) | PTFE (Teflon) — order code `FPA`; PTFE + nitrile wiper — `FPA**W` |
| Response time | 100 ms without frequent switching; 70 ms with S7 switch |
| Coil insulation class | Class H (180°C) |
| MTTF | 150 years (per ISO 13849) |
| Mounting standard | ISO 4401-03 / NFPA D-03 / ANSI B93.7M-D03 |

### Physical dimensions

| Model | Length | Width | Height |
|---|---|---|---|
| Dual solenoid | 219.43 mm (8.639 in) | 47.0 mm (1.850 in) | 23.50 mm (0.925 in) |
| Single solenoid | 155.72 mm (6.130 in) | 47.0 mm (1.850 in) | 23.50 mm (0.925 in) |

Weight: ~1.8–2.2 kg (see `hydraulics_engineer.md` DG4V-3 vs DG4V-5 table).

### Bolt kit / seal kit

- Interface Seal Kit number for DG4V-3: 02-141871 (for use with D03 / ISO 4401-03)
- Bolt Kit for ISO 4401-03: M6 × 1.0 fixing bolts, four-bolt pattern

---

## 7. Spool Position Indicator — S7 DC Model

From page 6 of the guide (Spool Position Indicator Models):

Available for spool/spring arrangement types: 0A, 0B, 2A, 2B, 22A, 23A, 35A, 52B, 3B, 6B.

**S7 specifications:**

| Parameter | Value |
|---|---|
| Supply voltage | 20–32 VDC |
| Reverse polarity protection | Yes |
| Output | 2 outputs with alternating function — PNP |
| Max output load | ≤400 mA; Duty Ratio 100% |
| Short circuit protection | Yes |
| Hysteresis | ≤0.05 mm |
| Electrical connector | M12 × 1 4-pole |
| Thermal shift | ≤±0.1 mm |
| Protection class | IP65 DIN 40050 |
| Vibration | 0–500Hz; max 20g |
| Shock | Max 50g |
| EMC | DIN EN 61000-6-1/2/3/4 |

**M12 4-pin wiring (S7):**
- Pin 1: Normal Closed (NC)
- Pin 2: Normal Open (NO)
- Pin 3: 0V (negative)
- Pin 4: +Supply (positive)

The S7 is ordered as an add-on code in the ordering code string (suffix `S7`).
It is documented in a separate catalogue (ref: AF458770480968en-000101).

---

## 8. Surge Suppression (Section 12)

From page 14 of the guide. The three codes at section 12 of the ordering code:

### D1 — Diode positive bias
- Diode in parallel with coil, positive bias
- Significantly reduces voltage spike on disconnect
- Recommended for DC coils
- May slightly increase drop-out time

### D2 — Diode negative bias
- Diode in parallel with coil, negative bias
- Same function as D1 but reversed polarity

### D7 — Transorb type
- Transorb (bidirectional voltage-clamping) device in parallel with coil
- Faster drop-out than diode suppressors
- Higher clamping voltage than diode — spike not fully suppressed
- See Page 14 of the guide for circuit details

### Blank — no surge suppressor
- Default if no surge code specified
- Not recommended for DC coils — voltage transients can damage PLC outputs

---

## 9. DG4V-3 Datasheet Layout (Page-by-Page Guide)

Understanding which pages contain which content is essential for vision-based extraction:

| Page | Content | Extraction method |
|---|---|---|
| 1 | Cover — product photo, series name, ISO standard | Text / Vision |
| 2 | **General description**, key features, performance claims, bolt kit, seal kit info | Vision — text is embedded in formatted layout |
| 3 | **Model code diagram** — the complete ordering code with numbered position boxes and option tables | **Vision CRITICAL** — entire page is vector graphics; virtually no extractable text |
| 4 | **Functional symbols** — spool types with ISO hydraulic diagrams; M vs VM style explanation; spool 8 note | **Vision CRITICAL** — ISO symbols are graphics; only spool codes in text |
| 5 | **Operating data** — pressure, flow, temperature, viscosity, response times, MTTF | Vision — data table is graphic; some text fragments extractable |
| 6 | Spool position indicator (S7 model) specs and wiring; EMC compliance info | Mixed text/vision |
| 7 | **Performance data** — flow vs pressure drop graphs for AC (50Hz, 60Hz) and DC models | Vision — graphs only; small flow-rate table partially readable |
| 8 | **Performance data** — pressure drop table by spool type; ● = available | Vision — table is graphic |
| 9 | **Installation dimensions** — ISO 4400 DIN connector models, double and single solenoid | Vision — dimensional drawings |
| 10 | Installation dimensions — flying lead ("F" type) and conduit box connectors | Vision |
| 11 | Installation dimensions — M12 connector type; electrical schematic | Vision |
| 12 | **Electrical plugs and connectors** — U/U1/U6, KU, KUP4, KUP5, X5 drawings; DIN 43650 detail | Vision + some text |
| 13 | **Terminal and lights** — wiring for "F" type coils; PA configuration; Insta-Plug wiring | Vision + text |
| 14 | **NFPA connector**, 3-pin and 5-pin connectors; surge suppression circuits (diode, Transsorb) | Vision + text |

**Page 3 is the most important page** — it contains the complete ordering code.
This page is 100% vector graphics with no extractable text.  VISION is required.

**Page 4 is critical for spool classification** — the ISO symbols define the canonical
spool function.  Use spool codes from this page combined with the reference in
`hydraulics_engineer.md` section 2 to assign canonical patterns.

---

## 10. Performance Data Summary

### DC solenoid maximum flow rates (from page 7, Graph 3 table)

Performance based on full power solenoid coils warm, operating at 90% rated voltage,
with mineral oil at 36 cSt (168.6 SUS) and specific gravity 0.87.

The guide shows performance curves with pressure drop (bar / psi) vs flow (l/min / USgpm).
Read from the graphs rather than relying on a single number — the maximum flow at any
given pressure drop is spool-type dependent.

Approximate maximum flows (DC, 315 bar rated pressure):
- Most closed-centre spools (2C, 2A, etc.): up to 80 l/min at reasonable pressure drop
- Open centre spools (0A, 0C): higher flow, lower restriction

### AC performance

AC coils at 50Hz: lower flow capacity than DC (different solenoid force profile).
AC coils at 60Hz: slightly higher than 50Hz.
Use the guide's specific AC graphs (pages 7–8) for accurate flow ratings.

---

## 11. DG4V-3-Specific Extraction Failure Modes

### 1. Entire guide is vector graphics — no text extraction is possible

Nearly every page is a scanned / vector-drawn document.  PyMuPDF typically extracts
fewer than 50 characters per page from the DG4V-3 guide.  **Vision is required for
every substantive page.**  Do NOT assume any content is in the text layer.  Any
`ordering_code.py` call on this guide must use the vision path.

### 2. Section 14 (tank pressure codes) are misread as bar values

The four codes at section 14 (`4`, `6`, `7`, `8`) are **specification codes, not
bar values**.  Code `7` means DC high performance at 207 bar (3000 psi), not 7 bar.
Specifically:
- `4` = 70 bar (1000 psi), **X5 coil type only** ▲
- `6` = 207 bar (3000 psi), AC high performance (incl. S7 switch)
- `7` = 207 bar (3000 psi), DC high performance (incl. S7 switch) ← **most common**
- `8` = 160 bar (2300 psi), AC high performance lower tank port rating (no X5 restriction)
- ▲ The X5-only restriction applies solely to code 4

If vision only extracts `4` from the diagram (which shows one representative value),
the injection constant `_DANFOSS_TANK_PRESSURE_OPTIONS` will add codes 6, 7, 8.
The `maps_to_value` for each code must be the T-port pressure in bar (70, 207, 207, 160),
not the code digit itself.

### 3. Spool 8 requires BOTH VM style AND design 61

Page 4 states clearly: "*Spool 8 only offered in VM style nomenclature."  The ordering
code footnotes on page 3 state design 61 is mandatory for spool 8*.
If EITHER constraint is missing from extraction, invalid codes will be generated.
Always check that both constraints appear in the `"constraints"` array.

### 4. M vs VM is NOT always visible as a separate segment

In many extracted codes the `V` in `VM` is easily confused with a separate segment.
It is part of the `style` segment:
- `M` = standard style
- `VM` = reversed solenoid style
Do NOT create a separate segment for the `V` alone unless following Option B described
in section 2.  If you create a separate `V` segment, ensure the constraint correctly
assembles to `VM` not `V` followed by a separate `M`.

### 5. Connector and voltage appear concatenated (no separator)

In `DG4V-3-2C-M-U-H7-60`, the connector `U` and voltage `H7` have no separator
between them.  Do NOT insert a hyphen or slash.  The code_template must show:
`{series}-{spool}-{style}-{connector}{voltage}-{design}` — connector and voltage
as adjacent placeholders without separator.

### 6. Seal prefix appears BEFORE the series, not after a hyphen

Wrong template: `DG4V-3-{seal}-{spool}-...`
Correct template: `{seal}DG4V-3-{spool}-...`
The `FPA` prefix, when present, immediately precedes `DG4V-3` with no hyphen:
`FPA-DG4V-3-...` (the hyphen is between FPA and DG4V, not a code separator).

### 7. Spool code suffix `L` is NOT lapped spool

`L` in `2AL`, `22AL` = **left-hand build** (spring-offset direction mirrors).
It does NOT mean lapped or lipped spool — this is a common misconception from older
catalogues.  LH and RH builds are NOT interchangeable.

### 8. Performance data page (page 7) contains spool codes NOT in the ordering code

The performance graphs on pages 7–8 include spool designations such as `4C`, `5C`,
`1BL`, `0BL1` that are historical/appendix codes not listed in the main ordering code
table on page 3.  Do NOT add these as ordering code options unless they appear on
page 3 or page 4.  They are performance comparison references, not orderable products
in the standard range.

### 9. Position 4 (`M`) appears fixed in the Danfoss-branded guide

In the Vickers-branded version of this guide, position 4 would show `C`, `A`, or `D`
(spring type) as variable options.  In the **Danfoss-branded** guide (this file),
position 4 is always `M` and is `is_fixed = true`.  The spring type is embedded in
the spool code itself (suffix C, A, AL, B, BL, N).  Do NOT extract `C`, `A`, `D` as
separate segment options for position 4 in this guide.

### 10. S7 switch ordering code reference

The spool position indicator S7 is not fully specified in this user guide — its detailed
specification lives in a separate catalogue (AF458770480968en-000101).  When extracting,
include `S7` as an option in the `spool_switch` segment but do not attempt to
extract full S7 specs from this guide.

### 11. MA and CCC coil variants

Page 3 notes: "MA and CCC approved coil — Details refer user guide
AF459779399265zh-000101."  These are special coil certifications (Maritime Authority,
China Compulsory Certificate) that generate additional ordering code variants.  They
are documented in a separate guide; do NOT attempt to extract them from this document.

---

## 12. Cross-Reference Summary for Matching

The DG4V-3 and Bosch Rexroth 4WE6 share ISO 4401-03 bolt pattern and are direct
body interchanges, but coils, seals, and internal cartridges are not cross-compatible.

For spool-function matching via canonical patterns, see `hydraulics_engineer.md`
section 4 (Rexroth↔Danfoss cross-reference table).  Key matches:
- Rexroth `E` → Danfoss `2C` (both BLOCKED)
- Rexroth `G` → Danfoss `8C` (both TANDEM)
- Rexroth `H` → Danfoss `0C` (**NOT** Danfoss `H` — false friend)
- Rexroth `J` → Danfoss `6C` (both FLOAT)

For the matching pipeline, the `coil_connector` field maps as:
- Danfoss `U` / `U1` / `U6` ↔ Rexroth `K4` (both DIN 43650)
- Danfoss `KUP5` ↔ Rexroth `N9` / Deutsch (similar sealing class)
