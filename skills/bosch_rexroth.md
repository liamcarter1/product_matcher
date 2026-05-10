# Bosch Rexroth Hydraulic Valve Domain Knowledge

Manufacturer-specific extraction guidance for Bosch Rexroth directional control valves.
This file is loaded in addition to `hydraulics_engineer.md` whenever the manufacturer
is identified as "Bosch Rexroth". The universal spool taxonomy, canonical patterns, and
cross-reference tables live in `hydraulics_engineer.md` — this file covers Rexroth-specific
ordering code structure, datasheet layout, and extraction failure modes.

---

## 1. 4WE6 Ordering Code (NG6 / CETOP 3 / ISO 4401-03)

### Template

```
4 WE 6 {spool} {design} / {style} {voltage} {override} {connector}
│  │  │    │       │    │    │        │          │           │
01 02 03   04      05  SEP  06       07         08          09
```

No spaces in the actual code. Separator `/` is a literal character that MUST appear in
the code_template: `{01}{02}{03}{04}{05}/{06}{07}{08}{09}`

### Position definitions

| Pos | Field | is_fixed | Typical options |
|---|---|---|---|
| 01 | `num_ports` | **true** | `4` |
| 02 | `valve_type` | **true** | `WE` |
| 03 | `valve_size` | **true** | `6` |
| 04 | `spool_type` | false | A, B, C, D, E, E1, E2, E3, F, G, H, J, L, M, P, Q, R, T, U, V, W, Y — plus soft-shift variants: E73, G73, H73, J73 (see note below) |
| 05 | `design_generation` | false | `6X` (current), `62`, `63`, `64` — all interchangeable per RE 23178 |
| SEP | literal `/` | — | always present |
| 06 | `electrical_style` | false | `E` = central plug (one connector for both solenoids); `A` = individual solenoid-a connector; `B` = individual solenoid-b connector |
| 07 | `coil_voltage` | false | `G12` (12VDC), `G24` (24VDC), `G48` (48VDC), `W110` (110VAC 50/60Hz), `W230` (230VAC 50/60Hz) |
| 08 | `manual_override` | false | `N9` (mechanical push), `NW` (waterproof push), `NH` (hand lever), `NF` (foot pedal), `""` (no override) |
| 09 | `coil_connector` | false | `K4` (DIN 43650 Form A metric), `K20` (DIN 43650 Form A inch), `""` (bare leads / standard) |

### Worked examples

| Code | Breakdown |
|---|---|
| `4WE6E62/EG24N9K4` | 4-way, WE, NG6, spool-E (BLOCKED), design-62, central-plug, 24VDC, mech-override, DIN-43650-A |
| `4WE6G6X/EG24N9K4` | Same but spool-G (TANDEM) and design any-6X gen |
| `4WE6H6X/AG24` | Spool-H (OPEN), individual solenoid-a connector, 24VDC, no override, bare leads |
| `4WE6E6X/EW230` | Spool-E (BLOCKED), central plug, 230VAC, no override |
| `4WE6E736X/EG24N9K4` | Spool-E73 (BLOCKED soft-shift), all else same |

### Soft-shift and transition variants in position 04

The soft-shift suffix `73` and transition variants `E1/E2/E3` are part of the
spool-type segment (position 04), NOT separate positions. Extract them as
additional options in the spool_type segment:

```json
{"code": "E",   "maps_to_field": "spool_type", "maps_to_value": "E"},
{"code": "E73", "maps_to_field": "spool_type", "maps_to_value": "E73"},
{"code": "E1",  "maps_to_field": "spool_type", "maps_to_value": "E1"},
{"code": "G",   "maps_to_field": "spool_type", "maps_to_value": "G"},
{"code": "G73", "maps_to_field": "spool_type", "maps_to_value": "G73"}
```

Do NOT create a separate position for `73` or for `1/2/3` after the spool letter.

### Design generation (position 05)

`6X`, `62`, `63`, `64` are all interchangeable per RE 23178. Extract all as
options with is_fixed=false. `6X` means "any current sub-generation" — it is a
valid orderable code, not a wildcard to be excluded.

### The "/" separator rule

The slash **MUST** appear in the code_template as a literal character.
`separator_before` for position 06 (electrical_style) = `"/"`.
The code_template must be: `{01}{02}{03}{04}{05}/{06}{07}{08}{09}`
NOT: `{01}{02}{03}{04}{05}{06}{07}{08}{09}` (missing slash).

---

## 2. 4WE10 Ordering Code (NG10 / CETOP 5 / ISO 4401-05)

Same structure as 4WE6 with these differences:

| Field | 4WE6 | 4WE10 |
|---|---|---|
| Position 03 (valve_size) | `6` | `10` |
| Max pressure (shared_specs) | 315 bar | 350 bar |
| Max flow (shared_specs) | 80 l/min | 120 l/min |
| Weight | ~1.9 kg | ~3.5 kg |
| Mounting | ISO 4401-03 | ISO 4401-05 |

Template: `{01}{02}{03}{04}{05}/{06}{07}{08}{09}`
Where position 01=`4`, 02=`WE`, 03=`10`.

The spool letter codes (position 04) are the **same set** as 4WE6.
The electrical options (positions 06–09) are the **same set** as 4WE6.

---

## 3. 4WRE / 4WREE Proportional Series Ordering Code (NG6)

Proportional directional control valves — different structure from the on/off 4WE series.

### Template

```
4 WRE {E} 6 {spool} {flow} {design} / {voltage} {electronics}
01  02   03 04  05     06     07    SEP    08         09
```

`{01}{02}{03}{04}{05}{06}{07}/{08}{09}`

| Pos | Field | Options |
|---|---|---|
| 01 | `num_ports` | `4` (fixed) |
| 02 | `valve_type` | `WRE` (fixed) |
| 03 | `electronics` | `E` = with integrated electronics; blank = without |
| 04 | `valve_size` | `6` (fixed for this series) |
| 05 | `spool_type` | `E` = standard (most common), others available |
| 06 | `flow_class` | `04` (4 l/min at 10 bar), `08` (8 l/min), `16` (16 l/min), `32` (32 l/min) |
| 07 | `design_generation` | `3XV`, `3X`, `2X` |
| SEP | `/` | literal separator |
| 08 | `coil_voltage` | `24` (24VDC — note: no G prefix here, just the number) |
| 09 | `option_code` | `A1` (standard), others |

Example: `4WREE6E04-3XV/24A1`

Key difference from 4WE series: no `/E` or `/G` style prefix — the voltage
is a bare number after the slash. The proportional series does NOT use the
G12/G24/W230 voltage coding of the on/off series.

---

## 4. Electrical Options — Four-Page Structure in RE Datasheets

Rexroth RE datasheets split electrical options across **four separate sections**:

| Section | Style | Current type | Voltage options |
|---|---|---|---|
| DC individual | A or B | DC | G12, G24, G48 |
| DC central plug | E | DC | G12, G24, G48 |
| AC individual | A or B | AC | W110, W230 |
| AC central plug | E | AC | W110, W230 |

**Critical extraction rule**: ALL four sections must be read to capture the full
set of electrical options. Each section is a matrix:
- Rows = voltage code (G12, G24, G48, W110, W230)
- Columns = connector type (K4, K20, bare, etc.)
- Cells contain a checkmark (✓) if the combination is available, or are blank if not

A checkmark matrix means: available combinations only. Do NOT generate
combinations that have no checkmark.

If you can only read one or two sections (e.g. vision only captured pages 3–4
of a 12-page datasheet), WARN that the electrical options may be incomplete.

---

## 5. Rexroth RE Datasheet Layout

RE datasheets follow a consistent page structure. Knowing this helps locate content:

| Pages | Content | Extraction method |
|---|---|---|
| 1 | Product overview, key specs, features | Text |
| 2 | Ordering code diagram (numbered boxes) | **Vision** — diagram is vector graphics |
| 3–4 | Spool type sub-table with ISO symbols | **Vision** — symbols are graphics |
| 5–6 | Electrical options (DC individual + central) | Text or Vision (checkmark matrix) |
| 7 | Electrical options (AC individual + central) | Text or Vision |
| 8 | Technical data (pressure, flow, temps) | Text |
| 9–10 | Characteristic curves | Vision (graphs, not extractable as data) |
| 11–12 | Dimensions and mounting (ISO 4401 bolt pattern) | Text/Vision |

**Page 2 is the most important for ordering code extraction** — it shows the
visual box diagram with numbered positions. This page is almost always
graphics-only with no extractable text. Use VISION on page 2.

**Pages 3–4 for spool types** — ISO 1219 diagrams. The text on these pages may
contain only the spool codes (A, B, C, ...) with no descriptions. Use the
canonical spool reference table in hydraulics_engineer.md to fill in descriptions.

---

## 6. Rexroth-Specific Extraction Failure Modes

### 1. Missing the "/" separator in the template
The most common structural error. The template MUST include the literal `/`.
Check: does the assembled code contain a slash? `4WE6E62/EG24N9K4` — yes.
If your template produces `4WE6E62EG24N9K4` (no slash) — the template is wrong.

### 2. Merging style + voltage into one segment
Wrong: single option `EG24` as one code for one position.
Correct: two separate positions — style (`E`) then voltage (`G24`).
The merge prevents generating `EG12`, `EG48`, `EW230` as distinct combinations.

### 3. Extracting only DC options, missing AC
If only pages 5–6 are captured, AC options (W110, W230) will be missing.
Ensure pages 7 (AC section) is included in the extraction.

### 4. Treating 73 suffix as a separate spool-modifier position
Wrong: position 04 = spool letter only (`E`), position 04a = variant suffix (`73`).
Correct: `E73` is a single option in position 04, same as `E`.

### 5. Treating 6X as a wildcard and omitting it
`6X` IS a valid orderable code — include it as a real option in the design_generation
segment alongside `62`, `63`. Do NOT treat it as a regex or placeholder.

### 6. Omitting the "no override" (blank) option for position 08
The most common configuration has no manual override. Include `""` (empty string)
as an option in the override segment. Without it, every generated code will have
an override suffix even when the customer wants none.

### 7. Omitting the "no connector" (blank/bare leads) option for position 09
Some Rexroth valves ship with bare leads (no DIN plug). Include `""` as an option
alongside `K4`. Without it, every generated code includes `K4` unnecessarily.

### 8. Confusing the spool-type page with the ordering code page
The spool-type page (pages 3–4) shows ISO symbols, not the full ordering code.
Extract spool codes FROM this page but do NOT use it to define the ordering code
template — the template is on page 2.
