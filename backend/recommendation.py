# project-root/backend/recommendation.py
import pandas as pd
import numpy as np
import os
import itertools
from collections import Counter, defaultdict

# New imports for ML evaluation/training
try:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn import metrics
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

class CollegeRecommender:
    """
    Recommendation engine with associative rule mining boost.

    - Program is mandatory (lenient matching: exact -> contains)
    - Rank is mandatory for filtering (Predicted Closing Rank >= user_rank)
    - Other filters optional (stream, quota, category, district, year)
    - CTC / scores are used for ordering (not as hard exclusion unless user requests)
    - No global "return everything" fallback — if filters cannot be satisfied, returns no-data.
    - Association rules mined from historical rows (Program/Stream/Quota/Category/District) and saved to csv.
      Rules are used to boost candidates that match rule consequents when user-provided antecedents match.
    """

    RULES_FILENAME = 'associates_rules.csv'  # stored under data_root_dir/csv/

    def __init__(self, data_root_dir="."):
        self.data_root_dir = os.path.abspath(data_root_dir)
        self.dataframes = {}
        self._load_all_data()
        self._prepare_master_rank_df()
        self._prepare_quality_data()

        # ensure rules exist (generate if not)
        try:
            self._ensure_rules()
        except Exception as e:
            print("Warning: association rules generation failed:", e)

        if getattr(self, 'merged_df', pd.DataFrame()).empty:
            print("⚠️ WARNING: Master rank data is empty. Recommendations will fail.")

    # ---------------------
    # IO & loading helpers
    # ---------------------
    def _get_file_path(self, file_name):
        return os.path.join(self.data_root_dir, 'csv', file_name)

    def _load_all_data(self):
        file_names = [
            'college.csv', 'rank_2021.csv', 'rank_2022.csv', 'rank_2023.csv',
            'rank_2024.csv', 'rank_2025.csv', 'placement.csv', 'reviews.csv'
        ]
        for file_name in file_names:
            try:
                df_name = file_name.replace('.csv', '')
                file_path = self._get_file_path(file_name)
                self.dataframes[df_name] = pd.read_csv(file_path, dtype='object')
            except FileNotFoundError:
                print(f"Warning: File {file_name} not found and skipped.")
                continue
            except Exception as e:
                print(f"Error loading {file_name}: {e}")

    # -------------------------
    # Rank / master data prep
    # -------------------------
    def _prepare_master_rank_df(self):
        combined_rank_data = []
        REQUIRED_COLUMNS = ['Year', 'Round', 'Institute', 'Program', 'Stream', 'Seat Type', 'Quota', 'Category', 'Opening Rank', 'Closing Rank']

        for df_name, df in self.dataframes.items():
            if df_name.startswith('rank_20'):
                df = df.copy()
                try:
                    year = int(df_name.split('_')[1])
                except Exception:
                    year = pd.NA
                df['Year'] = year

                if 'Seat Type' not in df.columns:
                    df['Seat Type'] = 'N/A'

                if 'Stream' in df.columns and not df['Stream'].empty:
                    df['Stream'] = df['Stream'].fillna('').astype(str).str.replace(
                        r'B\.E/B\.Tech.*|B\.E/B\.Arch.*|B\.Tech.*',
                        'b.e/b. tech',
                        regex=True
                    )

                for col in REQUIRED_COLUMNS:
                    if col not in df.columns:
                        df[col] = pd.NA

                combined_rank_data.append(df[REQUIRED_COLUMNS])

        if combined_rank_data:
            master_df = pd.concat(combined_rank_data, ignore_index=True)
            for col in ['Opening Rank', 'Closing Rank']:
                master_df[col] = pd.to_numeric(master_df[col], errors='coerce')

            # Standardize key text columns for reliable filtering
            text_cols_to_clean = ['Institute', 'Program', 'Stream', 'Quota', 'Category', 'Seat Type']
            for col in text_cols_to_clean:
                if col in master_df.columns:
                    master_df[col] = master_df[col].astype(str).str.strip().str.lower().replace('nan', '').fillna('')
                else:
                    master_df[col] = ''
        else:
            self.merged_df = pd.DataFrame(columns=REQUIRED_COLUMNS + ['District'])
            self.master_rank_df = pd.DataFrame()
            return

        # Merge district info from college.csv if available
        if 'college' in self.dataframes:
            college_df_for_merge = self.dataframes['college'][['Institute', 'District']].copy()
            college_df_for_merge['Institute'] = college_df_for_merge['Institute'].astype(str).str.strip().str.lower().replace('nan', '').fillna('')
            college_df_for_merge['District'] = college_df_for_merge['District'].astype(str).str.strip().str.lower().replace('nan', '').fillna('')
            self.merged_df = pd.merge(master_df, college_df_for_merge, on='Institute', how='left')
            if 'District' in self.merged_df.columns:
                self.merged_df['District'] = self.merged_df['District'].astype(str).str.strip().str.lower().replace('nan', '').fillna('')
        else:
            self.merged_df = master_df
            if 'District' not in self.merged_df.columns:
                self.merged_df['District'] = pd.NA

        self.master_rank_df = master_df

    # -------------------------
    # Quality (placement/review)
    # -------------------------
    def _prepare_quality_data(self):
        # College details
        self.full_college_df = self.dataframes.get('college', pd.DataFrame()).copy().rename(columns={'logo_image_url': 'logo_image'})
        if not self.full_college_df.empty:
            cols = ['Institute', 'District', 'Location', 'Website', 'logo_image', 'Picture']
            for c in cols:
                if c not in self.full_college_df.columns:
                    self.full_college_df[c] = ''
            self.full_college_df = self.full_college_df.reindex(columns=cols)
        if 'Institute' in self.full_college_df.columns:
            self.full_college_df['Institute'] = self.full_college_df['Institute'].astype(str).str.strip().str.lower().replace('nan', '').fillna('')
        if 'District' in self.full_college_df.columns:
            self.full_college_df['District'] = self.full_college_df['District'].astype(str).str.strip().str.lower().replace('nan', '').fillna('')

        # Placement data
        placement_df = self.dataframes.get('placement', pd.DataFrame()).copy()
        self.full_placement_df = pd.DataFrame()
        self.placement_max_ctc = pd.DataFrame()
        if not placement_df.empty:
            if 'Institute' in placement_df.columns:
                placement_df['Institute'] = placement_df['Institute'].astype(str).str.strip().str.lower().replace('nan', '').fillna('')
            if 'Program' in placement_df.columns:
                placement_df['Program'] = placement_df['Program'].astype(str).str.strip().str.lower().replace('nan', '').fillna('')
            placement_df = placement_df.rename(columns={'top_recruiters': 'top recruiter', 'job_titles': 'job_title', 'inst_rank': 'institute_rank'})
            ctc_cols = ['average_ctc', 'median_ctc', 'highest_ctc']
            for col in ctc_cols:
                if col in placement_df.columns:
                    placement_df[col] = pd.to_numeric(placement_df[col], errors='coerce')
            if 'average_ctc' in placement_df.columns:
                try:
                    self.placement_max_ctc = placement_df.groupby(['Institute', 'Program'])['average_ctc'].max().reset_index().rename(columns={'average_ctc': 'Max Average CTC'})
                except Exception:
                    self.placement_max_ctc = pd.DataFrame()
            try:
                agg_funcs = {}
                for col in ['average_ctc', 'median_ctc', 'highest_ctc']:
                    if col in placement_df.columns:
                        agg_funcs[col] = 'max'
                for col in ['top recruiter', 'job_title', 'institute_rank']:
                    if col in placement_df.columns:
                        agg_funcs[col] = 'first'
                grp_cols = [c for c in ['Institute', 'Program'] if c in placement_df.columns]
                if agg_funcs and grp_cols:
                    self.full_placement_df = placement_df.groupby(grp_cols).agg(agg_funcs).reset_index()
            except Exception:
                self.full_placement_df = pd.DataFrame()

        # Reviews data
        reviews_df = self.dataframes.get('reviews', pd.DataFrame()).copy()
        self.full_reviews_df = pd.DataFrame()
        self.reviews_avg_for_filter = pd.DataFrame()
        if not reviews_df.empty:
            reviews_df = reviews_df.rename(columns={'college_name': 'Institute'})
            if 'Institute' in reviews_df.columns:
                reviews_df['Institute'] = reviews_df['Institute'].astype(str).str.strip().str.lower().replace('nan', '').fillna('')
            review_cols = ['rating', 'sentiment_score', 'mess_score', 'professor_score', 'campus_score', 'placements_score', 'infrastructure_score', 'overall_aspect_score']
            for col in review_cols:
                if col in reviews_df.columns:
                    reviews_df[col] = pd.to_numeric(reviews_df[col], errors='coerce')
            present_review_cols = [c for c in review_cols if c in reviews_df.columns]
            if present_review_cols:
                self.full_reviews_df = reviews_df.groupby('Institute').agg({c: 'mean' for c in present_review_cols}).reset_index()
                if 'placements_score' in self.full_reviews_df.columns:
                    self.full_reviews_df = self.full_reviews_df.rename(columns={'placements_score': 'placement_score'})
                review_filter_cols = [c for c in ['mess_score', 'professor_score', 'campus_score', 'placements_score', 'infrastructure_score', 'overall_aspect_score'] if c in reviews_df.columns]
                if review_filter_cols:
                    tmp = reviews_df[['Institute'] + review_filter_cols].groupby('Institute')[review_filter_cols].mean().reset_index()
                    tmp = tmp.rename(columns={'placements_score': 'placements_score_filter', 'overall_aspect_score': 'overall_aspect_score_filter'})
                    self.reviews_avg_for_filter = tmp

        # Combine quality metrics
        try:
            if not self.placement_max_ctc.empty and not self.reviews_avg_for_filter.empty:
                self.combined_quality_df = pd.merge(self.placement_max_ctc, self.reviews_avg_for_filter, on='Institute', how='left')
            elif not self.placement_max_ctc.empty:
                self.combined_quality_df = self.placement_max_ctc.copy()
            elif not self.reviews_avg_for_filter.empty:
                self.combined_quality_df = self.reviews_avg_for_filter.copy()
            else:
                self.combined_quality_df = pd.DataFrame()
        except Exception:
            self.combined_quality_df = pd.DataFrame()

    # -------------------------
    # Metadata getters
    # -------------------------
    def get_unique_programs(self):
        return sorted(self.master_rank_df['Program'].dropna().unique().tolist()) if not self.master_rank_df.empty else []

    def get_unique_streams(self):
        return sorted(self.master_rank_df['Stream'].dropna().unique().tolist()) if not self.master_rank_df.empty else []

    def get_unique_quotas(self):
        return sorted(self.master_rank_df['Quota'].dropna().unique().tolist()) if not self.master_rank_df.empty else []

    def get_unique_categories(self):
        return sorted(self.master_rank_df['Category'].dropna().unique().tolist()) if not self.master_rank_df.empty else []

    def get_unique_locations(self):
        return sorted(self.merged_df['District'].dropna().unique().tolist()) if not self.merged_df.empty else []

    # -------------------------
    # Lenient matching helpers
    # -------------------------
    def _clean_user_input(self, s):
        return s.strip().lower() if isinstance(s, str) else ''

    def _lenient_match(self, series, value):
        """Try exact match first; if none found, try contains. Series expected to be cleaned already."""
        if value == '':
            return pd.Series([True] * len(series), index=series.index)
        exact = series == value
        if exact.any():
            return exact
        return series.str.contains(value, na=False)

    # -------------------------
    # Core prediction logic
    # -------------------------
    def _predict_top_colleges_rank_only(self, program, stream='', quota='', category='', district='', target_year=2026):
        df = self.merged_df.copy()

        program = self._clean_user_input(program)
        stream = self._clean_user_input(stream)
        quota = self._clean_user_input(quota)
        category = self._clean_user_input(category)
        district = self._clean_user_input(district)

        for col in ['Program', 'Stream', 'Quota', 'Category', 'District', 'Institute']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower().replace('nan', '').fillna('')
            else:
                df[col] = ''

        if program and 'tfw' in program and not category:
            category = 'tuition fee waiver'

        def apply_filters(df_in, prog, strm, qta, cat, dist):
            tmp = df_in.copy()
            if prog:
                exact = tmp[tmp['Program'] == prog].copy()
                if not exact.empty:
                    tmp = exact
                else:
                    contains = tmp[tmp['Program'].str.contains(prog, na=False)].copy()
                    tmp = contains
            if strm:
                t = tmp[tmp['Stream'] == strm].copy()
                if t.empty:
                    t = tmp[tmp['Stream'].str.contains(strm, na=False)].copy()
                if not t.empty:
                    tmp = t
            if qta:
                t = tmp[tmp['Quota'] == qta].copy()
                if t.empty:
                    t = tmp[tmp['Quota'].str.contains(qta, na=False)].copy()
                if not t.empty:
                    tmp = t
            if dist:
                t = tmp[tmp['District'].fillna('') == dist].copy()
                if t.empty:
                    t = tmp[tmp['District'].str.contains(dist, na=False)].copy()
                if not t.empty:
                    tmp = t
            if cat:
                t = tmp[tmp['Category'] == cat].copy()
                if t.empty:
                    t = tmp[tmp['Category'].str.contains(cat, na=False)].copy()
                if not t.empty:
                    tmp = t
            return tmp

        filtered_df = apply_filters(df, program, stream, quota, category, district)
        if filtered_df.empty:
            filtered_df = apply_filters(df, program, stream, quota, '', district)
        if filtered_df.empty:
            filtered_df = apply_filters(df, program, stream, '', '', district)
        if filtered_df.empty:
            filtered_df = apply_filters(df, program, stream, '', '', '')
        if filtered_df.empty:
            filtered_df = apply_filters(df, program, '', '', '', '')
        if filtered_df.empty and program:
            filtered_df = df[df['Program'].str.contains(program, na=False)].copy()

        if filtered_df.empty:
            return pd.DataFrame()

        final_ranks = filtered_df.sort_values('Round', ascending=False).drop_duplicates(
            subset=['Year', 'Institute', 'Stream', 'Quota', 'Category'], keep='first'
        )

        grouping_cols = ['Institute', 'Program', 'Stream', 'Quota', 'Category']
        historical_years = final_ranks['Year'].dropna().unique() if 'Year' in final_ranks.columns else []
        if target_year in historical_years:
            result_df = final_ranks[final_ranks['Year'] == target_year]
            if not result_df.empty:
                top_colleges = result_df.sort_values(by='Closing Rank', ascending=True)
                return top_colleges[['Institute', 'Program', 'Stream', 'Seat Type', 'Quota', 'Category', 'Opening Rank', 'Closing Rank']].rename(columns={'Closing Rank': 'Predicted Closing Rank'})

        prediction_results = []
        grouped_data = final_ranks.groupby(grouping_cols)
        for name, group in grouped_data:
            valid_ranks_group = group.dropna(subset=['Closing Rank'])
            if len(valid_ranks_group) >= 1:
                sorted_group = valid_ranks_group.sort_values(by='Year', ascending=False)
                recent = sorted_group['Closing Rank'].head(2).tolist()
                recent = [float(x) for x in recent if pd.notna(x)]
                if len(recent) == 0:
                    continue
                predicted_rank = float(np.mean(recent))
                predicted_rank = max(1.0, predicted_rank)

                latest_data = group.sort_values(by='Year', ascending=False).iloc[0]
                latest_opening_rank = latest_data['Opening Rank'] if pd.notna(latest_data['Opening Rank']) else np.nan
                latest_seat_type = latest_data['Seat Type'] if 'Seat Type' in latest_data.index else ''

                prediction_results.append({
                    'Institute': name[0], 'Program': name[1], 'Stream': name[2], 'Quota': name[3],
                    'Category': name[4], 'Predicted Closing Rank': predicted_rank,
                    'Opening Rank': latest_opening_rank, 'Seat Type': latest_seat_type
                })

        pred_df = pd.DataFrame(prediction_results)
        if pred_df.empty:
            return pd.DataFrame()
        return pred_df

    # -------------------------
    # Scoring / ordering helpers
    # -------------------------
    def _filter_top_colleges_by_metrics(self, df, min_ctc, min_placements_score=0):
        filtered_df = df.copy()
        if min_ctc > 0 and 'Max Average CTC' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Max Average CTC'] >= min_ctc]
        if min_placements_score > 0 and 'placements_score_filter' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['placements_score_filter'] >= min_placements_score]

        if filtered_df.empty:
            return pd.DataFrame()

        sort_cols = []
        if 'Max Average CTC' in filtered_df.columns:
            sort_cols.append('Max Average CTC')
        if 'overall_aspect_score_filter' in filtered_df.columns:
            sort_cols.append('overall_aspect_score_filter')
        sort_cols.append('Predicted Closing Rank')

        ascending = [False if c in ['Max Average CTC', 'overall_aspect_score_filter'] else True for c in sort_cols]

        final_display = filtered_df.sort_values(by=sort_cols, ascending=ascending).drop_duplicates(
            subset=['Institute', 'Program', 'Stream', 'Quota', 'Category'], keep='first'
        ).head(10)

        return final_display

    def _finalize_table(self, ranked_filtered_df):
        final_df = ranked_filtered_df[['Institute', 'Program', 'Stream', 'Seat Type', 'Quota', 'Category', 'Opening Rank', 'Predicted Closing Rank']].copy()

        if not getattr(self, 'full_college_df', pd.DataFrame()).empty:
            final_df = pd.merge(final_df, self.full_college_df, on='Institute', how='left')
        else:
            final_df['District'] = ''
            final_df['Location'] = ''
            final_df['Website'] = ''
            final_df['logo_image'] = ''
            final_df['Picture'] = ''

        if not getattr(self, 'full_placement_df', pd.DataFrame()).empty:
            merge_on = ['Institute', 'Program'] if 'Program' in self.full_placement_df.columns else ['Institute']
            final_df = pd.merge(final_df, self.full_placement_df, on=merge_on, how='left')
        if not getattr(self, 'full_reviews_df', pd.DataFrame()).empty:
            final_df = pd.merge(final_df, self.full_reviews_df, on='Institute', how='left')

        all_cols_to_select = [
            'Institute', 'Program', 'Stream', 'Seat Type', 'Quota', 'Category',
            'Opening Rank', 'Predicted Closing Rank',
            'District', 'Location', 'Website', 'logo_image', 'Picture',
            'average_ctc', 'median_ctc', 'highest_ctc', 'top recruiter', 'job_title', 'institute_rank',
            'rating', 'sentiment_score', 'mess_score', 'professor_score', 'campus_score', 'placement_score',
            'infrastructure_score', 'overall_aspect_score'
        ]

        for c in all_cols_to_select:
            if c not in final_df.columns:
                final_df[c] = ''

        final_df = final_df.reindex(columns=all_cols_to_select)
        final_df = final_df.rename(columns={'Predicted Closing Rank': 'Closing Rank'})

        numeric_cols = ['Opening Rank', 'Closing Rank', 'average_ctc', 'median_ctc', 'highest_ctc',
                        'institute_rank', 'rating', 'sentiment_score', 'mess_score', 'professor_score',
                        'campus_score', 'placement_score', 'infrastructure_score', 'overall_aspect_score']

        for col in numeric_cols:
            if col in final_df.columns:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0)

        for col in final_df.columns:
            if col not in numeric_cols:
                final_df[col] = final_df[col].fillna('')

        return final_df.head(10)

    # -------------------------
    # Association rule mining
    # -------------------------
    def _ensure_rules(self):
        """Ensure rules CSV exists; otherwise generate rules from merged_df."""
        rules_path = self._get_file_path(self.RULES_FILENAME)
        # create csv dir if missing
        csv_dir = os.path.dirname(rules_path)
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)

        if os.path.exists(rules_path):
            try:
                rules_df = pd.read_csv(rules_path, dtype='object')
                # store in memory
                self.assoc_rules_df = rules_df
                return
            except Exception:
                pass

        # generate rules
        rules_df = self._generate_association_rules()
        try:
            rules_df.to_csv(rules_path, index=False)
        except Exception as e:
            print("Warning: failed to save association rules to csv:", e)
        self.assoc_rules_df = rules_df

    def _generate_association_rules(self, min_support=0.02, max_itemset_size=3):
        """
        Simple Apriori-like miner:
        - Build transactions from merged_df rows using attributes: Program, Stream, Quota, Category, District
        - Items are strings like "program=computer science & engineering"
        - Returns DataFrame of rules: antecedent (semicolon-separated), consequent (single item), support, confidence, lift
        """
        df = getattr(self, 'merged_df', pd.DataFrame()).copy()
        if df.empty:
            return pd.DataFrame(columns=['antecedent', 'consequent', 'support', 'confidence', 'lift'])

        attributes = ['Program', 'Stream', 'Quota', 'Category', 'District']
        transactions = []
        for _, row in df.iterrows():
            items = set()
            for a in attributes:
                if a in row and pd.notna(row[a]) and str(row[a]).strip() != '':
                    val = str(row[a]).strip().lower()
                    items.add(f"{a.lower()}={val}")
            if items:
                transactions.append(sorted(items))

        N = len(transactions)
        if N == 0:
            return pd.DataFrame(columns=['antecedent', 'consequent', 'support', 'confidence', 'lift'])

        # count frequent itemsets for sizes 1..max_itemset_size
        itemset_counts = Counter()
        for t in transactions:
            # generate combinations up to max size
            for k in range(1, max_itemset_size + 1):
                for comb in itertools.combinations(t, k):
                    itemset_counts[frozenset(comb)] += 1

        # filter by min_support
        frequent_itemsets = {iset: cnt for iset, cnt in itemset_counts.items() if (cnt / N) >= min_support}

        # compute support map
        support_map = {iset: cnt / N for iset, cnt in frequent_itemsets.items()}

        # generate rules: for each frequent itemset with size>=2, consider all splits where consequent size=1
        rules = []
        for iset in frequent_itemsets:
            if len(iset) < 2:
                continue
            items_list = list(iset)
            for consequent_item in items_list:
                antecedent = frozenset(iset - {consequent_item})
                consequent = frozenset([consequent_item])
                if antecedent in support_map and consequent in support_map:
                    sup_ab = support_map[iset]
                    sup_a = support_map[antecedent]
                    sup_b = support_map[consequent]
                    # confidence and lift
                    confidence = sup_ab / sup_a if sup_a > 0 else 0.0
                    lift = confidence / sup_b if sup_b > 0 else 0.0
                    rules.append({
                        'antecedent': ';'.join(sorted(list(antecedent))),
                        'consequent': ';'.join(sorted(list(consequent))),
                        'support': round(sup_ab, 6),
                        'confidence': round(confidence, 6),
                        'lift': round(lift, 6)
                    })

        rules_df = pd.DataFrame(rules)
        # sort by confidence desc then support desc
        if not rules_df.empty:
            rules_df = rules_df.sort_values(by=['confidence', 'support'], ascending=[False, False]).reset_index(drop=True)
        else:
            rules_df = pd.DataFrame(columns=['antecedent', 'consequent', 'support', 'confidence', 'lift'])
        return rules_df

    # -------------------------
    # Rule-boosting helper
    # -------------------------
    def _compute_boosts_from_rules(self, candidates_df, user_filters):
        """
        Given candidates_df (with Institute, Program, Stream, Quota, Category, District columns cleaned)
        and user_filters dict (program/stream/quota/category/district cleaned),
        compute a boost score per candidate using matching rules.

        Strategy:
        - For each rule whose antecedent is fully present in user_filters (i.e., user provided matching values),
          check the rule's consequent (single item). If the candidate matches the consequent, add boost += confidence * support.
        - Return dict: institute_program_key -> boost_score
        """
        boosts = defaultdict(float)
        rules_df = getattr(self, 'assoc_rules_df', pd.DataFrame())
        if rules_df is None or rules_df.empty:
            return boosts

        # Pre-build candidate attribute sets for quick check
        cand_attrs = {}
        for _, row in candidates_df.iterrows():
            key = (row.get('Institute', ''), row.get('Program', ''))
            items = set()
            for a in ['Program', 'Stream', 'Quota', 'Category', 'District']:
                v = row.get(a, '')
                if pd.notna(v) and str(v).strip() != '':
                    items.add(f"{a.lower()}={str(v).strip().lower()}")
            cand_attrs[key] = items

        # Build user-provided item set from filters (only include filters the user provided non-empty)
        user_items = set()
        for k, v in user_filters.items():
            if v and str(v).strip() != '':
                user_items.add(f"{k.lower()}={str(v).strip().lower()}")

        # For each rule, check if antecedent subset of user_items
        for _, r in rules_df.iterrows():
            antecedent_items = set(filter(None, [s.strip() for s in (r.get('antecedent') or '').split(';')]))
            consequent_items = set(filter(None, [s.strip() for s in (r.get('consequent') or '').split(';')]))
            if not antecedent_items:
                continue
            # antecedent must be fully present in user_items to apply
            if antecedent_items.issubset(user_items):
                for cand_key, cand_set in cand_attrs.items():
                    # if candidate contains all consequent items, apply boost
                    if consequent_items.issubset(cand_set):
                        try:
                            boost_value = float(r.get('confidence', 0.0)) * float(r.get('support', 0.0))
                        except Exception:
                            boost_value = 0.0
                        boosts[cand_key] += boost_value
        return boosts

    # -------------------------
    # Helper to build group-level predictions across entire dataset for ML training/eval
    # -------------------------
    def _build_group_predictions_all(self):
        """
        Build a grouped DataFrame across entire master_rank_df similar to the per-program predictor.
        Output columns: Institute, Program, Stream, Quota, Category, Predicted Closing Rank, Latest Closing Rank, Opening Rank
        """
        df = getattr(self, 'master_rank_df', pd.DataFrame()).copy()
        if df.empty:
            return pd.DataFrame()

        grouping_cols = ['Institute', 'Program', 'Stream', 'Quota', 'Category']
        df['Closing Rank'] = pd.to_numeric(df['Closing Rank'], errors='coerce')
        prediction_results = []
        grouped = df.groupby(grouping_cols)

        for name, group in grouped:
            valid_ranks_group = group.dropna(subset=['Closing Rank'])
            if len(valid_ranks_group) >= 1:
                sorted_group = valid_ranks_group.sort_values(by='Year', ascending=False)
                recent = sorted_group['Closing Rank'].head(2).tolist()
                recent = [float(x) for x in recent if pd.notna(x)]
                if len(recent) == 0:
                    continue
                predicted_rank = float(np.mean(recent))
                predicted_rank = max(1.0, predicted_rank)

                latest_row = group.sort_values(by='Year', ascending=False).iloc[0]
                latest_closing = latest_row['Closing Rank'] if pd.notna(latest_row['Closing Rank']) else np.nan
                latest_opening = latest_row['Opening Rank'] if 'Opening Rank' in latest_row.index and pd.notna(latest_row['Opening Rank']) else np.nan

                prediction_results.append({
                    'Institute': name[0], 'Program': name[1], 'Stream': name[2], 'Quota': name[3],
                    'Category': name[4],
                    'Predicted Closing Rank': predicted_rank,
                    'Latest Closing Rank': latest_closing,
                    'Opening Rank': latest_opening
                })

        return pd.DataFrame(prediction_results)

    # -------------------------
    # Evaluate heuristic vs Decision Tree and return chosen model info
    # -------------------------
    def _evaluate_and_train_ml(self, user_rank_threshold):
        """
        Build a dataset and evaluate two approaches:
        - Heuristic: predicted closing rank >= user_rank_threshold (binary)
        - Decision Tree: trained to predict the same target (Latest Closing Rank >= user_rank_threshold)
        Prints accuracy, precision, recall, f1 for both to console.
        Returns:
            chosen_method: 'heuristic' or 'decision_tree'
            dt_model: trained DecisionTreeClassifier or None
            label_encoders: dict of LabelEncoders used (for later encoding) or {}
        """
        # Prepare grouped data
        grouped_df = self._build_group_predictions_all()
        if grouped_df.empty:
            print("ML Eval: no grouped historical data available. Skipping ML evaluation.")
            return 'heuristic', None, {}

        # Truth label: was the latest closing rank >= user_rank_threshold (consistent with earlier filter logic)
        grouped_df = grouped_df.dropna(subset=['Latest Closing Rank'])
        grouped_df['Latest Closing Rank'] = pd.to_numeric(grouped_df['Latest Closing Rank'], errors='coerce')
        grouped_df = grouped_df.dropna(subset=['Latest Closing Rank']).copy()
        if grouped_df.empty:
            print("ML Eval: no entries with latest closing rank. Skipping ML evaluation.")
            return 'heuristic', None, {}

        grouped_df['true_label'] = (grouped_df['Latest Closing Rank'] >= float(user_rank_threshold)).astype(int)

        # Heuristic predictions (using predicted closing rank)
        grouped_df['heuristic_pred'] = (grouped_df['Predicted Closing Rank'] >= float(user_rank_threshold)).astype(int)

        # Print heuristic metrics
        try:
            h_acc = metrics.accuracy_score(grouped_df['true_label'], grouped_df['heuristic_pred'])
            h_prec = metrics.precision_score(grouped_df['true_label'], grouped_df['heuristic_pred'], zero_division=0)
            h_rec = metrics.recall_score(grouped_df['true_label'], grouped_df['heuristic_pred'], zero_division=0)
            h_f1 = metrics.f1_score(grouped_df['true_label'], grouped_df['heuristic_pred'], zero_division=0)
        except Exception:
            h_acc = h_prec = h_rec = h_f1 = 0.0

        print(f"[Heuristic] accuracy: {h_acc:.4f}, precision: {h_prec:.4f}, recall: {h_rec:.4f}, f1: {h_f1:.4f}")

        # If sklearn not available, return heuristic
        if not SKLEARN_AVAILABLE:
            print("scikit-learn not available. Using heuristic for ranking.")
            return 'heuristic', None, {}

        # Build features for DT: encode Program, Stream, Quota, Category, Institute (label encode)
        features = ['Institute', 'Program', 'Stream', 'Quota', 'Category']
        X = grouped_df[features].fillna('').astype(str).apply(lambda col: col.str.strip().str.lower())
        label_encoders = {}
        X_enc = pd.DataFrame()
        for col in X.columns:
            le = LabelEncoder()
            try:
                X_enc[col] = le.fit_transform(X[col])
            except Exception:
                # if only one unique value, LabelEncoder may still work, but handle gracefully
                X_enc[col] = 0
                le = None
            label_encoders[col] = le

        # Optionally include predicted closing rank as numeric feature
        X_enc['predicted_closing_rank'] = grouped_df['Predicted Closing Rank'].astype(float).fillna(grouped_df['Predicted Closing Rank'].median())

        y = grouped_df['true_label'].astype(int)

        # train/test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=42, stratify=y if len(y.unique())>1 else None)
        except Exception:
            X_train, X_test, y_train, y_test = X_enc, X_enc, y, y

        # Train Decision Tree
        try:
            dt = DecisionTreeClassifier(random_state=42, max_depth=6)
            dt.fit(X_train, y_train)
            y_pred = dt.predict(X_test)
            dt_acc = metrics.accuracy_score(y_test, y_pred)
            dt_prec = metrics.precision_score(y_test, y_pred, zero_division=0)
            dt_rec = metrics.recall_score(y_test, y_pred, zero_division=0)
            dt_f1 = metrics.f1_score(y_test, y_pred, zero_division=0)
        except Exception as e:
            print("Decision Tree training failed:", e)
            return 'heuristic', None, {}

        print(f"[DecisionTree] accuracy: {dt_acc:.4f}, precision: {dt_prec:.4f}, recall: {dt_rec:.4f}, f1: {dt_f1:.4f}")

        # Choose best by accuracy
        chosen = 'decision_tree' if dt_acc > h_acc else 'heuristic'
        print(f"Chosen model for further ranking: {chosen} (heuristic acc {h_acc:.4f} vs dt acc {dt_acc:.4f})")

        return chosen, dt if chosen == 'decision_tree' else None, label_encoders if chosen == 'decision_tree' else {}

    # -------------------------
    # Public recommend method
    # -------------------------
    def recommend(self, user_rank, user_program, user_stream='', user_quota='', user_category='', user_location='', min_ctc=0, min_placements_score=0, target_year=2026):
        """
        Returns recommendations (status, message, data)
        - user_rank: numeric
        - user_program: mandatory
        - optional filters: user_stream, user_quota, user_category, user_location
        - min_ctc & min_placements_score can be used to reorder/filter when desired
        """

        ranked_predictions_df = self._predict_top_colleges_rank_only(
            program=user_program, stream=user_stream, quota=user_quota, category=user_category, district=user_location, target_year=target_year
        )

        if ranked_predictions_df.empty:
            return {'status': 'error', 'message': "No historical data found for the specified filters."}

        try:
            user_rank_val = float(user_rank)
        except Exception:
            return {'status': 'error', 'message': 'Invalid value for user_rank.'}

        ranked_results_df = ranked_predictions_df[ranked_predictions_df['Predicted Closing Rank'] >= user_rank_val].sort_values(by='Predicted Closing Rank', ascending=True)

        if ranked_results_df.empty:
            return {'status': 'error', 'message': f"No colleges found with a predicted closing rank ≥ {user_rank_val}. Consider increasing your expected rank (higher number) or broadening filters."}

        # Merge quality metrics
        if not getattr(self, 'combined_quality_df', pd.DataFrame()).empty:
            combined_filter_df = pd.merge(ranked_results_df, self.combined_quality_df, on=['Institute', 'Program'], how='left')
        else:
            combined_filter_df = ranked_results_df.copy()

        # ensure filter columns exist
        score_cols_filter = [
            'Max Average CTC', 'mess_score', 'professor_score', 'campus_score',
            'placements_score_filter', 'infrastructure_score', 'overall_aspect_score_filter'
        ]
        for col in score_cols_filter:
            if col not in combined_filter_df.columns:
                combined_filter_df[col] = 0
            combined_filter_df[col] = combined_filter_df[col].fillna(0)

        # Apply optional min_ctc/min_placements_score only if user requested (>0)
        if (min_ctc > 0 or min_placements_score > 0):
            final_filtered_results = self._filter_top_colleges_by_metrics(combined_filter_df, min_ctc=min_ctc, min_placements_score=min_placements_score)
        else:
            final_filtered_results = combined_filter_df

        if final_filtered_results.empty:
            # If filtering by CTC/score removed everything, return warning with empty data (you asked to remove "global fallback")
            return {'status': 'warning', 'message': "Quality filters removed all candidates. Try lowering min CTC/score or broadening other filters.", 'data': []}

        # Prepare final merged table
        final_table_candidates = self._finalize_table(final_filtered_results)

        # Build user filter items to match rules
        user_filters = {
            'program': self._clean_user_input(user_program),
            'stream': self._clean_user_input(user_stream),
            'quota': self._clean_user_input(user_quota),
            'category': self._clean_user_input(user_category),
            'district': self._clean_user_input(user_location)
        }

        # compute rule-based boosts
        # Note: final_table_candidates has Institute & Program columns in their cleaned (lowercase) form; ensure keys
        # Build a candidates DataFrame for the boosting function with cleaned textual columns
        cand_df_for_boost = final_table_candidates.copy()
        for col in ['Program', 'Stream', 'Quota', 'Category', 'District', 'Institute']:
            if col in cand_df_for_boost.columns:
                cand_df_for_boost[col] = cand_df_for_boost[col].astype(str).str.strip().str.lower().replace('nan', '').fillna('')
            else:
                cand_df_for_boost[col] = ''
        # compute boosts (map keyed by (Institute, Program))
        boosts = self._compute_boosts_from_rules(cand_df_for_boost, user_filters)

        # attach boost column and sort: higher boost first, then by Closing Rank ascending (better rank)
        def cand_key(row):
            return (row.get('Institute', ''), row.get('Program', ''))

        final_table_candidates['_boost'] = final_table_candidates.apply(lambda r: boosts.get(cand_key(r), 0.0), axis=1)

        # -------------------------
        # New ML evaluation/training step
        # -------------------------
        # Use the provided user_rank_val as the threshold for evaluation/training
        chosen_model, dt_model, label_encoders = self._evaluate_and_train_ml(user_rank_val)

        # If decision_tree chosen and model available, compute model probability per candidate and use as additional sort key
        if chosen_model == 'decision_tree' and dt_model is not None and SKLEARN_AVAILABLE:
            # Build features same as training
            def encode_feature_value(col, val):
                le = label_encoders.get(col)
                try:
                    if le is None:
                        return 0
                    return int(le.transform([str(val).strip().lower()])[0])
                except Exception:
                    # unseen label -> attempt to add via fit? we cannot change encoder; fallback to -1 or 0
                    return 0

            ml_probs = []
            for _, row in final_table_candidates.iterrows():
                inst = str(row.get('Institute', '')).strip().lower()
                prog = str(row.get('Program', '')).strip().lower()
                strm = str(row.get('Stream', '')).strip().lower()
                qta = str(row.get('Quota', '')).strip().lower()
                cat = str(row.get('Category', '')).strip().lower()
                pred_rank = float(row.get('Closing Rank', 0) if pd.notna(row.get('Closing Rank', None)) else 0)

                feat_inst = encode_feature_value('Institute', inst)
                feat_prog = encode_feature_value('Program', prog)
                feat_strm = encode_feature_value('Stream', strm)
                feat_qta = encode_feature_value('Quota', qta)
                feat_cat = encode_feature_value('Category', cat)

                X_row = pd.DataFrame([{
                    'Institute': feat_inst, 'Program': feat_prog, 'Stream': feat_strm, 'Quota': feat_qta, 'Category': feat_cat,
                    'predicted_closing_rank': pred_rank
                }])
                try:
                    prob = float(dt_model.predict_proba(X_row)[:, 1][0])
                except Exception:
                    try:
                        prob = float(dt_model.predict(X_row)[0])
                    except Exception:
                        prob = 0.0
                ml_probs.append(prob)

            final_table_candidates['_ml_prob'] = ml_probs
            # sort by boost desc, ml_prob desc, then Closing Rank asc
            final_table_candidates = final_table_candidates.sort_values(by=['_boost', '_ml_prob', 'Closing Rank'], ascending=[False, False, True])
            # remove helper ml column after sorting if you wish to keep output clean; but user asked to "use" model, not to remove, so leave it out of returned records
            final_table_candidates = final_table_candidates.drop(columns=['_ml_prob'])
        else:
            # Heuristic chosen (or sklearn not available). keep existing sorting: by _boost then Closing Rank asc
            final_table_candidates = final_table_candidates.sort_values(by=['_boost', 'Closing Rank'], ascending=[False, True])

        # drop helper column after sorting
        final_table_candidates = final_table_candidates.drop(columns=['_boost'])

        # Return top 10 (already limited in _finalize_table but ensure)
        result_list = final_table_candidates.head(10).to_dict('records')

        return {'status': 'success', 'message': 'Top college recommendations based on rank, quality and association-rule boosting:', 'data': result_list}
