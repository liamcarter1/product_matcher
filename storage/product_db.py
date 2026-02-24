"""
ProductMatchPro - SQLite Product Database
Stores structured product records, model code patterns, confirmed equivalents, and feedback.
Thread-safety: All write operations are serialised through a threading.Lock.
"""

import sqlite3
import json
import uuid
import threading
from pathlib import Path
from typing import Optional
from fuzzywuzzy import fuzz, process

from models import (
    HydraulicProduct, ScoreBreakdown, MatchResult,
    ModelCodePattern, SCORE_WEIGHTS, CONFIDENCE_THRESHOLD,
    SPEC_FIELDS, NUMERICAL_FIELDS, EXACT_MATCH_FIELDS,
)

DB_PATH = Path(__file__).parent.parent / "data" / "products.db"


class ProductDB:

    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path
        self._lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        self._migrate_schema()

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS products (
                id TEXT PRIMARY KEY,
                company TEXT NOT NULL,
                model_code TEXT NOT NULL,
                product_name TEXT DEFAULT '',
                category TEXT DEFAULT '',
                subcategory TEXT,
                max_pressure_bar REAL,
                max_flow_lpm REAL,
                valve_size TEXT,
                spool_type TEXT,
                num_positions INTEGER,
                num_ports INTEGER,
                actuator_type TEXT,
                coil_voltage TEXT,
                coil_type TEXT,
                coil_connector TEXT,
                port_size TEXT,
                port_type TEXT,
                mounting TEXT,
                mounting_pattern TEXT,
                body_material TEXT,
                seal_material TEXT,
                operating_temp_min_c REAL,
                operating_temp_max_c REAL,
                fluid_type TEXT,
                viscosity_range_cst TEXT,
                weight_kg REAL,
                displacement_cc REAL,
                speed_rpm_max REAL,
                bore_diameter_mm REAL,
                rod_diameter_mm REAL,
                stroke_mm REAL,
                extra_specs TEXT DEFAULT '{}',
                description TEXT DEFAULT '',
                source_document TEXT DEFAULT '',
                raw_text TEXT DEFAULT '',
                model_code_decoded TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_model_code ON products(model_code);
            CREATE INDEX IF NOT EXISTS idx_company ON products(company);
            CREATE INDEX IF NOT EXISTS idx_category ON products(category);
            CREATE INDEX IF NOT EXISTS idx_coil_voltage ON products(coil_voltage);
            CREATE INDEX IF NOT EXISTS idx_valve_size ON products(valve_size);

            CREATE TABLE IF NOT EXISTS model_code_patterns (
                id TEXT PRIMARY KEY,
                company TEXT NOT NULL,
                series TEXT NOT NULL,
                segment_position INTEGER,
                segment_name TEXT,
                code_value TEXT,
                decoded_value TEXT,
                maps_to_field TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_pattern_company
                ON model_code_patterns(company, series);

            CREATE TABLE IF NOT EXISTS confirmed_equivalents (
                id TEXT PRIMARY KEY,
                competitor_model_code TEXT NOT NULL,
                competitor_company TEXT NOT NULL,
                my_company_model_code TEXT NOT NULL,
                confirmed_by TEXT,
                confidence_override REAL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                query TEXT,
                competitor_model_code TEXT,
                my_company_model_code TEXT,
                confidence_score REAL,
                thumbs_up BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS synonyms (
                id TEXT PRIMARY KEY,
                term TEXT NOT NULL,
                canonical TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS series_cross_reference (
                id TEXT PRIMARY KEY,
                my_company_series TEXT NOT NULL,
                my_company_name TEXT NOT NULL DEFAULT 'Danfoss',
                competitor_series TEXT NOT NULL,
                competitor_company TEXT NOT NULL,
                product_type TEXT DEFAULT '',
                notes TEXT DEFAULT '',
                source_document TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(my_company_series, competitor_series)
            );

            CREATE INDEX IF NOT EXISTS idx_xref_competitor
                ON series_cross_reference(competitor_series, competitor_company);

            CREATE TABLE IF NOT EXISTS spool_type_reference (
                id TEXT PRIMARY KEY,
                series_prefix TEXT NOT NULL,
                manufacturer TEXT NOT NULL,
                spool_code TEXT NOT NULL,
                description TEXT DEFAULT '',
                center_condition TEXT DEFAULT '',
                solenoid_a_function TEXT DEFAULT '',
                solenoid_b_function TEXT DEFAULT '',
                canonical_pattern TEXT DEFAULT '',
                is_primary INTEGER DEFAULT 0,
                source TEXT DEFAULT 'manual',
                source_document TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(series_prefix, manufacturer, spool_code)
            );

            CREATE INDEX IF NOT EXISTS idx_spool_ref_series
                ON spool_type_reference(series_prefix, manufacturer);
        """)
        self.conn.commit()

    def _migrate_schema(self):
        """Add columns that may be missing in older databases. Safe and idempotent."""
        cursor = self.conn.execute("PRAGMA table_info(products)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        if "extra_specs" not in existing_columns:
            self.conn.execute(
                "ALTER TABLE products ADD COLUMN extra_specs TEXT DEFAULT '{}'"
            )
            self.conn.commit()

        # Migrate spool_type_reference if it exists but lacks is_primary
        cursor = self.conn.execute("PRAGMA table_info(spool_type_reference)")
        spool_columns = {row[1] for row in cursor.fetchall()}
        if spool_columns and "is_primary" not in spool_columns:
            self.conn.execute(
                "ALTER TABLE spool_type_reference ADD COLUMN is_primary INTEGER DEFAULT 0"
            )
            self.conn.commit()

    # ── Product CRUD ──────────────────────────────────────────────────

    def insert_product(self, product: HydraulicProduct) -> str:
        if not product.id:
            product.id = str(uuid.uuid4())
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO products (
                    id, company, model_code, product_name, category, subcategory,
                    max_pressure_bar, max_flow_lpm, valve_size, spool_type,
                    num_positions, num_ports, actuator_type, coil_voltage,
                    coil_type, coil_connector, port_size, port_type,
                    mounting, mounting_pattern, body_material, seal_material,
                    operating_temp_min_c, operating_temp_max_c, fluid_type,
                    viscosity_range_cst, weight_kg, displacement_cc,
                    speed_rpm_max, bore_diameter_mm, rod_diameter_mm, stroke_mm,
                    extra_specs,
                    description, source_document, raw_text, model_code_decoded
                ) VALUES (
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?,
                    ?, ?, ?, ?
                )
            """, (
                product.id, product.company, product.model_code,
                product.product_name, product.category, product.subcategory,
                product.max_pressure_bar, product.max_flow_lpm,
                product.valve_size, product.spool_type,
                product.num_positions, product.num_ports,
                product.actuator_type, product.coil_voltage,
                product.coil_type, product.coil_connector,
                product.port_size, product.port_type,
                product.mounting, product.mounting_pattern,
                product.body_material, product.seal_material,
                product.operating_temp_min_c, product.operating_temp_max_c,
                product.fluid_type, product.viscosity_range_cst,
                product.weight_kg, product.displacement_cc,
                product.speed_rpm_max, product.bore_diameter_mm,
                product.rod_diameter_mm, product.stroke_mm,
                json.dumps(product.extra_specs) if product.extra_specs else '{}',
                product.description, product.source_document, product.raw_text,
                json.dumps(product.model_code_decoded) if product.model_code_decoded else None,
            ))
            self.conn.commit()
        return product.id

    def get_product(self, product_id: str) -> Optional[HydraulicProduct]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM products WHERE id = ?", (product_id,))
        row = cursor.fetchone()
        if row:
            return self._row_to_product(row)
        return None

    def delete_product(self, product_id: str):
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM products WHERE id = ?", (product_id,))
            self.conn.commit()

    def get_all_products(self, company: Optional[str] = None) -> list[HydraulicProduct]:
        cursor = self.conn.cursor()
        if company:
            cursor.execute("SELECT * FROM products WHERE company = ?", (company,))
        else:
            cursor.execute("SELECT * FROM products")
        return [self._row_to_product(row) for row in cursor.fetchall()]

    def get_products_by_category(
        self, category: str, company: Optional[str] = None
    ) -> list[HydraulicProduct]:
        cursor = self.conn.cursor()
        if company:
            cursor.execute(
                "SELECT * FROM products WHERE category = ? AND company = ?",
                (category, company),
            )
        else:
            cursor.execute("SELECT * FROM products WHERE category = ?", (category,))
        return [self._row_to_product(row) for row in cursor.fetchall()]

    def get_product_counts(self) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT company, category, COUNT(*) as count
            FROM products GROUP BY company, category ORDER BY company, category
        """)
        return [dict(row) for row in cursor.fetchall()]

    def get_companies(self) -> list[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT company FROM products ORDER BY company")
        return [row["company"] for row in cursor.fetchall()]

    def get_all_model_codes(self, company: Optional[str] = None) -> list[str]:
        cursor = self.conn.cursor()
        if company:
            cursor.execute(
                "SELECT model_code FROM products WHERE company = ?", (company,)
            )
        else:
            cursor.execute("SELECT model_code FROM products")
        return [row["model_code"] for row in cursor.fetchall()]

    # ── Fuzzy Model Code Lookup ───────────────────────────────────────

    def fuzzy_lookup_model(
        self,
        partial_code: str,
        company: Optional[str] = None,
        limit: int = 5,
        threshold: int = 60,
    ) -> list[tuple[HydraulicProduct, float]]:
        """Fuzzy match a partial model code against the database.
        Returns list of (product, match_score) tuples sorted by score descending."""
        all_codes = self.get_all_model_codes(company)
        if not all_codes:
            return []

        upper_codes = [c.upper() for c in all_codes]
        partial_upper = partial_code.upper()

        # Use multiple scorers: token_sort_ratio for full matches,
        # partial_ratio for substring/partial code matches
        matches_token = process.extract(
            partial_upper, upper_codes,
            scorer=fuzz.token_sort_ratio, limit=limit * 2,
        )
        matches_partial = process.extract(
            partial_upper, upper_codes,
            scorer=fuzz.partial_ratio, limit=limit * 2,
        )

        # Also check prefix matches (common for partial model codes)
        prefix_matches = [
            (code, 95) for code in upper_codes
            if code.startswith(partial_upper)
        ]

        # Merge and deduplicate, keeping best score per code
        best_scores: dict[str, int] = {}
        for match_list in [matches_token, matches_partial, prefix_matches]:
            for match_tuple in match_list:
                code = match_tuple[0]
                score = match_tuple[1]
                if code in best_scores:
                    best_scores[code] = max(best_scores[code], score)
                else:
                    best_scores[code] = score

        results = []
        for matched_code, score in best_scores.items():
            if score < threshold:
                continue
            cursor = self.conn.cursor()
            if company:
                cursor.execute(
                    "SELECT * FROM products WHERE UPPER(model_code) = ? AND company = ?",
                    (matched_code, company),
                )
            else:
                cursor.execute(
                    "SELECT * FROM products WHERE UPPER(model_code) = ?",
                    (matched_code,),
                )
            for row in cursor.fetchall():
                results.append((self._row_to_product(row), score / 100.0))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    # ── Model Code Decoding ───────────────────────────────────────────

    def decode_model_code(self, model_code: str, company: str) -> dict:
        """Decode a model code into structured specs using stored patterns."""
        cursor = self.conn.cursor()
        # Find the series that matches the start of the model code
        cursor.execute(
            "SELECT DISTINCT series FROM model_code_patterns WHERE company = ?",
            (company,),
        )
        series_list = [row["series"] for row in cursor.fetchall()]

        matching_series = None
        for series in series_list:
            if model_code.upper().startswith(series.upper()):
                matching_series = series
                break

        if not matching_series:
            return {}

        # Get all patterns for this series
        cursor.execute(
            """SELECT * FROM model_code_patterns
               WHERE company = ? AND series = ?
               ORDER BY segment_position""",
            (company, matching_series),
        )
        patterns = [dict(row) for row in cursor.fetchall()]

        # Split model code into segments (by dash or known boundaries)
        remaining = model_code[len(matching_series):]
        segments = [s for s in remaining.replace("-", " ").split() if s]

        decoded = {"series": matching_series}
        for pattern in patterns:
            pos = pattern["segment_position"]
            if pos < len(segments):
                segment_val = segments[pos].upper()
                if segment_val == pattern["code_value"].upper():
                    decoded[pattern["segment_name"]] = pattern["decoded_value"]
                    decoded[f"_raw_{pattern['segment_name']}"] = segment_val
                    # Store the explicit field mapping for _apply_decoded_specs
                    if pattern.get("maps_to_field"):
                        decoded[f"_field_{pattern['segment_name']}"] = pattern["maps_to_field"]

        return decoded

    def insert_model_code_pattern(self, pattern: ModelCodePattern):
        with self._lock:
            cursor = self.conn.cursor()
            pattern_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT OR REPLACE INTO model_code_patterns
                (id, company, series, segment_position, segment_name,
                 code_value, decoded_value, maps_to_field)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern_id, pattern.company, pattern.series,
                pattern.segment_position, pattern.segment_name,
                pattern.code_value, pattern.decoded_value, pattern.maps_to_field,
            ))
            self.conn.commit()

    # ── Confirmed Equivalents ─────────────────────────────────────────

    def get_confirmed_equivalent(
        self, competitor_code: str
    ) -> Optional[HydraulicProduct]:
        """Check if there's a manually confirmed equivalent for this competitor code."""
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT my_company_model_code FROM confirmed_equivalents
               WHERE UPPER(competitor_model_code) = ?""",
            (competitor_code.upper(),),
        )
        row = cursor.fetchone()
        if row:
            my_code = row["my_company_model_code"]
            cursor.execute(
                "SELECT * FROM products WHERE UPPER(model_code) = ?",
                (my_code.upper(),),
            )
            product_row = cursor.fetchone()
            if product_row:
                return self._row_to_product(product_row)
        return None

    def insert_confirmed_equivalent(
        self,
        competitor_code: str,
        competitor_company: str,
        my_company_code: str,
        confirmed_by: str = "admin",
        confidence_override: float = 0.95,
        notes: str = "",
    ):
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO confirmed_equivalents
                (id, competitor_model_code, competitor_company,
                 my_company_model_code, confirmed_by, confidence_override, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()), competitor_code, competitor_company,
                my_company_code, confirmed_by, confidence_override, notes,
            ))
            self.conn.commit()

    # ── Feedback ──────────────────────────────────────────────────────

    def store_feedback(
        self,
        query: str,
        competitor_code: str,
        my_company_code: str,
        confidence: float,
        thumbs_up: bool,
    ):
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO feedback
                (id, query, competitor_model_code, my_company_model_code,
                 confidence_score, thumbs_up)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()), query, competitor_code,
                my_company_code, confidence, thumbs_up,
            ))
            self.conn.commit()

    def get_feedback(self, limit: int = 100) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]

    # ── Synonyms ──────────────────────────────────────────────────────

    def resolve_synonym(self, term: str) -> str:
        """Resolve a company/brand synonym to its canonical name."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT canonical FROM synonyms WHERE UPPER(term) = ?",
            (term.upper(),),
        )
        row = cursor.fetchone()
        return row["canonical"] if row else term

    def insert_synonym(self, term: str, canonical: str):
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO synonyms (id, term, canonical) VALUES (?, ?, ?)",
                (str(uuid.uuid4()), term, canonical),
            )
            self.conn.commit()

    # ── Series Cross-Reference ─────────────────────────────────────────

    def insert_series_cross_reference(
        self,
        my_company_series: str,
        competitor_series: str,
        competitor_company: str,
        product_type: str = "",
        notes: str = "",
        source_document: str = "",
        my_company_name: str = "Danfoss",
    ):
        """Insert or update a series-level cross-reference mapping."""
        with self._lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO series_cross_reference
                   (id, my_company_series, my_company_name, competitor_series,
                    competitor_company, product_type, notes, source_document)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), my_company_series, my_company_name,
                 competitor_series, competitor_company, product_type, notes,
                 source_document),
            )
            self.conn.commit()

    def lookup_series_by_competitor_prefix(
        self,
        competitor_model_code: str,
        competitor_company: str = None,
    ) -> list[dict]:
        """Find Danfoss series whose competitor_series is a prefix of the model code.

        Returns list of dicts with my_company_series, competitor_series, etc.
        Longest prefix match is returned first (most specific).
        """
        code_upper = competitor_model_code.upper().strip()
        cursor = self.conn.cursor()

        if competitor_company:
            cursor.execute(
                """SELECT * FROM series_cross_reference
                   WHERE UPPER(competitor_company) = ?
                   ORDER BY LENGTH(competitor_series) DESC""",
                (competitor_company.upper(),),
            )
        else:
            cursor.execute(
                """SELECT * FROM series_cross_reference
                   ORDER BY LENGTH(competitor_series) DESC""",
            )

        results = []
        for row in cursor.fetchall():
            prefix = row["competitor_series"].upper().strip()
            if code_upper.startswith(prefix):
                results.append(dict(row))

        return results

    def get_all_cross_references(self) -> list[dict]:
        """Return all series cross-reference mappings for admin display."""
        cursor = self.conn.execute(
            "SELECT * FROM series_cross_reference ORDER BY competitor_company, competitor_series"
        )
        return [dict(row) for row in cursor.fetchall()]

    def delete_cross_references_by_source(self, source_document: str):
        """Delete all cross-references from a specific source document."""
        with self._lock:
            self.conn.execute(
                "DELETE FROM series_cross_reference WHERE source_document = ?",
                (source_document,),
            )
            self.conn.commit()

    # ── Spool Type Reference ────────────────────────────────────────────

    def insert_spool_type_reference(
        self,
        series_prefix: str,
        manufacturer: str,
        spool_code: str,
        description: str = "",
        center_condition: str = "",
        solenoid_a_function: str = "",
        solenoid_b_function: str = "",
        canonical_pattern: str = "",
        is_primary: bool = False,
        source: str = "manual",
        source_document: str = "",
    ):
        """Insert or update a known spool type for a manufacturer/series."""
        with self._lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO spool_type_reference
                   (id, series_prefix, manufacturer, spool_code, description,
                    center_condition, solenoid_a_function, solenoid_b_function,
                    canonical_pattern, is_primary, source, source_document)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), series_prefix, manufacturer, spool_code,
                 description, center_condition, solenoid_a_function,
                 solenoid_b_function, canonical_pattern,
                 1 if is_primary else 0, source, source_document),
            )
            self.conn.commit()

    def get_spool_type_references(
        self,
        series_prefix: str = None,
        manufacturer: str = None,
    ) -> list[dict]:
        """Get known spool types, optionally filtered by series and/or manufacturer.

        Fuzzy prefix matching: stored 'DG4V' matches query 'DG4V-3' and vice versa.
        """
        cursor = self.conn.cursor()

        if manufacturer and series_prefix:
            # Fetch all rows for this manufacturer, then filter by fuzzy prefix
            cursor.execute(
                """SELECT * FROM spool_type_reference
                   WHERE UPPER(manufacturer) = ?
                   ORDER BY series_prefix, spool_code""",
                (manufacturer.upper(),),
            )
            query_upper = series_prefix.upper().strip()
            results = []
            for row in cursor.fetchall():
                stored_upper = row["series_prefix"].upper().strip()
                # Fuzzy: stored is prefix of query OR query is prefix of stored
                if query_upper.startswith(stored_upper) or stored_upper.startswith(query_upper):
                    results.append(dict(row))
            return results
        elif manufacturer:
            cursor.execute(
                """SELECT * FROM spool_type_reference
                   WHERE UPPER(manufacturer) = ?
                   ORDER BY series_prefix, spool_code""",
                (manufacturer.upper(),),
            )
            return [dict(row) for row in cursor.fetchall()]
        elif series_prefix:
            cursor.execute(
                "SELECT * FROM spool_type_reference ORDER BY series_prefix, spool_code",
            )
            query_upper = series_prefix.upper().strip()
            results = []
            for row in cursor.fetchall():
                stored_upper = row["series_prefix"].upper().strip()
                if query_upper.startswith(stored_upper) or stored_upper.startswith(query_upper):
                    results.append(dict(row))
            return results
        else:
            cursor.execute(
                "SELECT * FROM spool_type_reference ORDER BY manufacturer, series_prefix, spool_code",
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_spool_codes_for_series(
        self,
        series_prefix: str,
        manufacturer: str,
    ) -> list[str]:
        """Return just the spool code strings for a specific series/manufacturer.

        Used for prompt injection and post-extraction validation.
        """
        refs = self.get_spool_type_references(series_prefix, manufacturer)
        return sorted(set(r["spool_code"] for r in refs))

    def bulk_insert_spool_type_references(self, references: list[dict]) -> int:
        """Insert multiple spool type references at once.

        Skips duplicates silently (INSERT OR IGNORE).
        Returns count of newly inserted rows.
        """
        inserted = 0
        with self._lock:
            cursor = self.conn.cursor()
            for ref in references:
                try:
                    cursor.execute(
                        """INSERT OR IGNORE INTO spool_type_reference
                           (id, series_prefix, manufacturer, spool_code, description,
                            center_condition, solenoid_a_function, solenoid_b_function,
                            canonical_pattern, is_primary, source, source_document)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (str(uuid.uuid4()),
                         ref.get("series_prefix", ""),
                         ref.get("manufacturer", ""),
                         ref.get("spool_code", ""),
                         ref.get("description", ""),
                         ref.get("center_condition", ""),
                         ref.get("solenoid_a_function", ""),
                         ref.get("solenoid_b_function", ""),
                         ref.get("canonical_pattern", ""),
                         1 if ref.get("is_primary") else 0,
                         ref.get("source", "auto_confirmed"),
                         ref.get("source_document", "")),
                    )
                    if cursor.rowcount > 0:
                        inserted += 1
                except sqlite3.IntegrityError:
                    continue
            self.conn.commit()
        return inserted

    def delete_spool_type_reference(self, ref_id: str):
        """Delete a single spool type reference by id (prefix match)."""
        with self._lock:
            cursor = self.conn.cursor()
            # Try exact match first
            cursor.execute(
                "DELETE FROM spool_type_reference WHERE id = ?", (ref_id,)
            )
            if cursor.rowcount == 0:
                # Try prefix match
                cursor.execute(
                    "DELETE FROM spool_type_reference WHERE id LIKE ?",
                    (ref_id + "%",),
                )
            self.conn.commit()

    def get_primary_spool_codes(
        self,
        series_prefix: str,
        manufacturer: str,
    ) -> list[str]:
        """Return spool codes marked as primary (main runners) for a series.

        Uses the same fuzzy prefix matching as get_spool_type_references.
        Returns empty list if no spools are marked primary (meaning: use all).
        """
        refs = self.get_spool_type_references(series_prefix, manufacturer)
        primary_codes = [r["spool_code"] for r in refs if r.get("is_primary")]
        return sorted(set(primary_codes))

    def update_spool_type_primary(self, ref_id: str, is_primary: bool):
        """Toggle the is_primary flag on a spool type reference."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE spool_type_reference SET is_primary = ? WHERE id = ?",
                (1 if is_primary else 0, ref_id),
            )
            if cursor.rowcount == 0:
                # Try prefix match
                cursor.execute(
                    "UPDATE spool_type_reference SET is_primary = ? WHERE id LIKE ?",
                    (1 if is_primary else 0, ref_id + "%"),
                )
            self.conn.commit()

    def set_all_spools_primary(
        self,
        manufacturer: str,
        spool_codes: list[str],
        series_prefix: str = None,
    ):
        """Mark specific spool codes as primary for a manufacturer."""
        with self._lock:
            for code in spool_codes:
                if series_prefix:
                    self.conn.execute(
                        """UPDATE spool_type_reference SET is_primary = 1
                           WHERE UPPER(spool_code) = ?
                             AND UPPER(manufacturer) = ?
                             AND UPPER(series_prefix) = ?""",
                        (code.upper(), manufacturer.upper(), series_prefix.upper()),
                    )
                else:
                    self.conn.execute(
                        """UPDATE spool_type_reference SET is_primary = 1
                           WHERE UPPER(spool_code) = ?
                             AND UPPER(manufacturer) = ?""",
                        (code.upper(), manufacturer.upper()),
                    )
            self.conn.commit()

    def clear_all_spools_primary(
        self,
        manufacturer: str,
        series_prefix: str = None,
    ):
        """Clear all primary flags for a manufacturer (optionally scoped to series)."""
        with self._lock:
            if series_prefix:
                self.conn.execute(
                    """UPDATE spool_type_reference SET is_primary = 0
                       WHERE UPPER(manufacturer) = ? AND UPPER(series_prefix) = ?""",
                    (manufacturer.upper(), series_prefix.upper()),
                )
            else:
                self.conn.execute(
                    """UPDATE spool_type_reference SET is_primary = 0
                       WHERE UPPER(manufacturer) = ?""",
                    (manufacturer.upper(),),
                )
            self.conn.commit()

    # ── Spec Comparison ───────────────────────────────────────────────

    def spec_comparison(
        self,
        competitor: HydraulicProduct,
        candidate: HydraulicProduct,
        semantic_score: float = 0.0,
    ) -> tuple[float, ScoreBreakdown]:
        """Compare two products spec-by-spec and compute a confidence score."""

        breakdown = ScoreBreakdown()

        # Category match (gate condition)
        if competitor.category and candidate.category:
            breakdown.category_match = (
                1.0 if competitor.category.lower() == candidate.category.lower() else 0.0
            )
        else:
            breakdown.category_match = 0.5  # Unknown

        # Core specs
        breakdown.pressure_match = self._numerical_match(
            competitor.max_pressure_bar, candidate.max_pressure_bar
        )
        breakdown.flow_match = self._numerical_match(
            competitor.max_flow_lpm, candidate.max_flow_lpm
        )
        breakdown.valve_size_match = self._exact_match(
            competitor.valve_size, candidate.valve_size
        )

        # Actuation & electrical
        breakdown.actuator_type_match = self._exact_match(
            competitor.actuator_type, candidate.actuator_type
        )
        breakdown.coil_voltage_match = self._exact_match(
            competitor.coil_voltage, candidate.coil_voltage
        )
        # Spool function match — prefer canonical pattern if available
        spool_score = self._exact_match(
            competitor.spool_type, candidate.spool_type
        )
        # Upgrade: if both have canonical spool patterns (flat key), use those for matching
        comp_pattern = (competitor.extra_specs or {}).get("canonical_spool_pattern")
        cand_pattern = (candidate.extra_specs or {}).get("canonical_spool_pattern")
        if comp_pattern and cand_pattern:
            spool_score = 1.0 if comp_pattern == cand_pattern else 0.0
        breakdown.spool_function_match = spool_score

        # Physical
        breakdown.port_match = self._exact_match(
            competitor.port_size, candidate.port_size
        )
        breakdown.mounting_match = self._exact_match(
            competitor.mounting, candidate.mounting
        )
        if breakdown.mounting_match < 1.0:
            # Try mounting_pattern as fallback
            breakdown.mounting_match = max(
                breakdown.mounting_match,
                self._exact_match(competitor.mounting_pattern, candidate.mounting_pattern),
            )

        # Materials
        breakdown.seal_material_match = self._exact_match(
            competitor.seal_material, candidate.seal_material
        )

        # Temperature range
        breakdown.temp_range_match = self._temp_range_match(
            competitor, candidate
        )

        # Semantic similarity
        breakdown.semantic_similarity = semantic_score

        # Spec coverage (includes extra_specs common keys)
        total_specs = 0
        covered_specs = 0
        for field in SPEC_FIELDS:
            comp_val = getattr(competitor, field, None)
            cand_val = getattr(candidate, field, None)
            if comp_val is not None or cand_val is not None:
                total_specs += 1
                if comp_val is not None and cand_val is not None:
                    covered_specs += 1

        # Bonus: count matching extra_specs keys (exclude _ prefixed internal keys)
        comp_extra = {k: v for k, v in (competitor.extra_specs or {}).items()
                      if not k.startswith("_")}
        cand_extra = {k: v for k, v in (candidate.extra_specs or {}).items()
                      if not k.startswith("_")}
        common_extra_keys = set(comp_extra.keys()) & set(cand_extra.keys())
        for key in common_extra_keys:
            total_specs += 1
            if str(comp_extra[key]).strip().lower() == str(cand_extra[key]).strip().lower():
                covered_specs += 1

        breakdown.spec_coverage = covered_specs / max(total_specs, 1)

        # Calculate weighted confidence score
        confidence = 0.0
        for field_name, weight in SCORE_WEIGHTS.items():
            score = getattr(breakdown, field_name, 0.0)
            confidence += score * weight

        # Category mismatch penalty: cap at 0.3
        if breakdown.category_match == 0.0:
            confidence = min(confidence, 0.3)

        confidence = max(0.0, min(1.0, confidence))

        return confidence, breakdown

    @staticmethod
    def _numerical_match(val_a: Optional[float], val_b: Optional[float]) -> float:
        """Score 0.0-1.0 based on how close two numerical values are."""
        if val_a is None or val_b is None:
            return 0.5  # Unknown, neutral score
        if val_a == 0 and val_b == 0:
            return 1.0
        max_val = max(abs(val_a), abs(val_b))
        if max_val == 0:
            return 1.0
        diff_ratio = abs(val_a - val_b) / max_val
        return max(0.0, 1.0 - diff_ratio)

    @staticmethod
    def _exact_match(val_a: Optional[str], val_b: Optional[str]) -> float:
        """String match, case-insensitive with fuzzy tolerance.

        Returns 1.0 for exact match, 0.75+ for near match (e.g. '24VDC' vs '24 VDC'),
        0.5 for unknown (one side is None), graded score for partial similarity,
        or 0.0 for completely different strings.
        """
        if val_a is None or val_b is None:
            return 0.5  # Unknown
        if not val_a or not val_b:
            return 0.5
        a = val_a.strip().lower()
        b = val_b.strip().lower()
        if a == b:
            return 1.0
        # Normalise: strip non-alphanumeric, collapse whitespace
        import re
        a_norm = re.sub(r'[^a-z0-9]', '', a)
        b_norm = re.sub(r'[^a-z0-9]', '', b)
        if a_norm == b_norm:
            return 0.95  # e.g. "24 VDC" vs "24VDC", "G 3/8" vs "G3/8"
        # Containment check — one value contains the other
        if a_norm in b_norm or b_norm in a_norm:
            return 0.75
        # Fuzzy token ratio for everything else
        ratio = fuzz.token_sort_ratio(a, b) / 100.0
        if ratio >= 0.8:
            return ratio
        return 0.0

    @staticmethod
    def _temp_range_match(
        competitor: HydraulicProduct, candidate: HydraulicProduct
    ) -> float:
        """Check if the candidate's temp range covers the competitor's range."""
        if (competitor.operating_temp_min_c is None
                or competitor.operating_temp_max_c is None
                or candidate.operating_temp_min_c is None
                or candidate.operating_temp_max_c is None):
            return 0.5  # Unknown

        min_ok = candidate.operating_temp_min_c <= competitor.operating_temp_min_c
        max_ok = candidate.operating_temp_max_c >= competitor.operating_temp_max_c
        if min_ok and max_ok:
            return 1.0
        elif min_ok or max_ok:
            return 0.5
        return 0.0

    # ── Helpers ────────────────────────────────────────────────────────

    def _row_to_product(self, row: sqlite3.Row) -> HydraulicProduct:
        d = dict(row)
        decoded = d.pop("model_code_decoded", None)
        d.pop("created_at", None)
        if decoded and isinstance(decoded, str):
            try:
                d["model_code_decoded"] = json.loads(decoded)
            except json.JSONDecodeError:
                d["model_code_decoded"] = None
        else:
            d["model_code_decoded"] = decoded

        # Deserialize extra_specs from JSON
        extra_raw = d.pop("extra_specs", None)
        if extra_raw and isinstance(extra_raw, str):
            try:
                d["extra_specs"] = json.loads(extra_raw)
            except json.JSONDecodeError:
                d["extra_specs"] = {}
        else:
            d["extra_specs"] = extra_raw if isinstance(extra_raw, dict) else {}

        return HydraulicProduct(**d)

    def close(self):
        self.conn.close()
