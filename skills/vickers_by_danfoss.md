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

## 2. DG4V-3 Ordering Code — Complete Structure

### Template

```
[seal] DG4V-3 - [spool] - [style] - [connector][voltage][surge][switch] - [design] - [tank]
 01      02       03         04          05        06      07     08          09         10
```

No spaces in the actual code.  `seal` is a prefix before `DG4V`, not after a hyphen.
The hyphens shown are literal code separators; `connector` and `voltage` are concatenated
with no separator between them.

Full extended example:
`DG4V-3-2C-M-UH7D-60-7`
`FPA-DG4V-3-8C-VM-UH7-61-7`

Abbreviated example (common, omits surge and tank when default):
`DG4V-3-2C-M-U-H7-60`

### Position definitions

| Pos | Segment name | is_fixed | Options |
|---|---|---|---|
| 01 | `seal_type` | false | blank = Viton/NBR (default, omitted); `FPA` = PTFE (Teflon); `FPA**W` = PTFE body seals + nitrile wiper seals |
| 02 | `series` | **true** | `DG4V-3` (always) |
| 03 | `spool_type` | false | See spool table in section 3 |
| 04 | `style` | false | `M` = standard (Sol. A on B-port side, P→A on activation); `VM` = reversed (Sol. A on A-port side, A→T on activation). **Spool 8* MUST use VM.** |
| 05 | `coil_connector` | false | See connector table in section 4 |
| 06 | `coil_voltage` | false | See voltage table in section 5 |
| 07 | `surge_suppressor` | false | blank = none (omitted); `D` or `D1` = diode (DC only, recommended); `ZT` = Transsorb/Zener (DC only) |
| 08 | `spool_switch` | false | blank = none (omitted); `S7` = DC spool position indicator (20–32VDC, M12 4-pin); `X5` = explosion-proof switch |
| 09 | `design_number` | false | `60` = standard; `61` = **mandatory for spool types beginning with 8**; `62`, `63` = other variants |
| 10 | `tank_back_pressure_bar` | false | `4`, `6`, `7`, `8` — T-port back-pressure rating in bar. **7 is the most common**; 4 is least common. Content is in a graphics diagram footnote — inject all four if only one is extracted. |

### Worked examples

| Code | Breakdown |
|---|---|
| `DG4V-3-2C-M-U-H7-60` | Standard, closed-centre, M-style, DIN connector, 24VDC, design 60 |
| `DG4V-3-8C-VM-U-H7-61` | Spool 8C, reversed solenoid (VM mandatory), 24VDC, design 61 (mandatory) |
| `FPA-DG4V-3-2C-M-KUP5-H7-60` | PTFE seals, Deutsch connector, 24VDC, design 60 |
| `DG4V-3-0A-M-U-A7-60` | Open centre, DIN connector, 110VAC 50Hz, design 60 |
| `DG4V-3-22A-M-U-H7-60-7` | Selector spool, 24VDC, design 60, 7 bar T-port |

### Spool 8-series mandatory constraints

**Both** of the following constraints MUST be returned in the `"constraints"` array
whenever spool types starting with `8` are present in the guide:

```json
[
  {
    "when_segment": "spool_type",
    "when_value_regex": "^8",
    "enforce_segment": "style",
    "enforce_value": "VM"
  },
  {
    "when_segment": "spool_type",
    "when_value_regex": "^8",
    "enforce_segment": "design_number",
    "enforce_value": "61"
  }
]
```

Page 4 of the guide states explicitly:
- "*Spool 8 only offered in VM style nomenclature"
- Design 61 required (stated in the ordering code footnotes)

Generating `DG4V-3-8C-M-U-H7-60` would be an invalid code that Danfoss cannot supply.

### "V" modifier in the model code (spool_8_modifier position)

The `hydraulics_engineer.md` section 1 documents a position 7 called `spool_8_modifier`
whose code is `"V"` for spool 8* and `""` (blank) for all others.  This `V` is the
same as the `VM` style — the `V` appears in the ordering code immediately before `M`,
making `VM` visible in the code.  When processing this guide, model it as two distinct
approaches:

**Option A** (single style segment, recommended): `style` segment with options `"M"` and
`"VM"`.  The `V` is embedded in the style option; no separate `spool_8_modifier`
segment is needed.

**Option B** (two segments, matching `hydraulics_engineer.md` position 7–8): separate
`spool_8_modifier` segment (options `""` and `"V"`) followed by `M` (fixed).
Constraint: spool 8* → modifier = `"V"`.

Either approach is valid; the assembled model code must contain `VM` for spool 8*
and `M` for all others.

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

## 5. Coil Voltage Codes

### Danfoss/Vickers DG4V-3 voltage codes

| Code | Voltage | Type | Frequency | Notes |
|---|---|---|---|---|
| **G7** | 12V | DC | 60 Hz wiring | Lower voltage DC |
| **H7** | 24V | DC | 60 Hz wiring | **Most common**; 0.8A steady state ≈ 26Ω coil resistance |
| **A7** | 110V | AC | 50Hz | Use AC-specific connector types |
| **B7** | 115V | AC | 60Hz | North American AC voltage |
| **C7** | 220V | AC | 50/60Hz | European mains |
| **D7** | 24V | DC | — | Alternative DC 24V variant (high-power coil) |

### Voltage code hierarchy (tier 1 + tier 2)

The single letter (`G`, `H`, `A`, `B`) is the voltage family; the digit (`7`) is the
frequency/wiring variant.  See `hydraulics_engineer.md` section 6 for the full hierarchy.

### AC coil characteristics (from page 5, Operating Data)

| Spec | At 50 Hz | At 60 Hz |
|---|---|---|
| Dual frequency power | 200 W | 250 W |
| Performance | Standard | Standard |

AC coils draw 4–5× steady-state current at inrush.  If the spool sticks, the coil
burns out at ~137 W vs ~22 W normal.  Recommend DC coils for reliability-critical
applications.

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

## 8. Surge Suppression

From page 14 of the guide:

### Diode suppressor (code `D` or `D1`)
- Diode in parallel with coil, positive bias
- DC voltage **only** — polarity dependent
- Significantly reduces voltage spike on valve disconnect
- Recommended for DC installations
- May slightly increase valve drop-out time

### Transsorb / Zener suppressor (code `ZT` or `Z`)
- Bidirectional voltage-clamping device in parallel with coil
- DC voltage only
- Faster drop-out than diode suppressor
- Higher clamping voltage than diode — spike not fully suppressed

### No suppressor (blank)
- Default if no surge code specified
- Only acceptable with AC valves (AC coils have built-in suppression in some designs)
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

### 2. Tank pressure options appear ONLY in a diagram footnote (page 3)

The ordering code diagram shows `4` as the representative tank pressure.  The footnote
table (also on page 3) lists all four options: **4, 6, 7, 8 bar**.  If vision misses
the footnote area, only `4` is extracted and the system generates 1/4 of the tank
pressure variants.  The injection constant `_DANFOSS_TANK_PRESSURE_OPTIONS` handles
this failure mode automatically — it will add the missing options if fewer than 4
are extracted.

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
