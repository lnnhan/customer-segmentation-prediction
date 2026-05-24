"""
Feature Engineering Module
Module này tạo ra các biến dẫn xuất phục vụ cho việc phân cụm (clustering) trong bộ dữ liệu khách hàng của Highlands Coffee
"""

import pandas as pd

df = pd.read_csv("merged_full.csv")
print("Loaded merged_full.csv:", df.shape)

class FeatureEngineering:
    def __init__(self, df):
        self.df = df
        self.keep = [
            "Independent Cafe",
            "Street / Half street coffee (including carts)",
            "Highlands Coffee",
            "Trung Nguyên"
        ]

        # Danh sách flag cho Highlands Coffee
        self.flag_map = {
            "Awareness": "Awareness_flag",
            "Spontaneous": "Spontaneous_flag",
            "Trial": "Trial_flag",
            "P3M": "P3M_flag",
            "P1M": "P1M_flag",
            "Weekly": "Weekly_flag",
            "Daily": "Daily_flag"
        }

    # -----------------------------
    # 1. Brand Grouping
    # -----------------------------
    def add_brand_grouping(self):
        self.df["TOM_Group"] = self.df["TOM"].apply(lambda x: x if x in self.keep else "Others")
        self.df["BUMO_Group"] = self.df["BUMO"].apply(lambda x: x if x in self.keep else "Others")
        self.df["BUMO_Previous_Group"] = self.df["BUMO_Previous"].apply(
            lambda x: x if x in self.keep else "Others"
        )
        self.df["MostFavourite_Group"] = self.df["MostFavourite"].apply(
            lambda x: x if x in self.keep else "Others"
        )
        return self

    # -----------------------------
    # 2. Brand Flags for Highlands Coffee
    # -----------------------------
    def add_brand_flags(self):
        for raw_col, flag_col in self.flag_map.items():
            self.df[flag_col] = (self.df[raw_col] == "Highlands Coffee").astype(int)
        return self
        
    # -----------------------------
    # 3. Loyalty Score (0–4)
    # -----------------------------
    def add_brand_loyalty(self):
        self.df["Brand_Loyalty"] = (
            (self.df["TOM_Group"] == self.df["BUMO_Group"]).astype(int) +
            (self.df["BUMO_Group"] == self.df["MostFavourite"]).astype(int) +
            (self.df["BUMO_Group"] == self.df["BUMO_Previous_Group"]).astype(int) +
            (self.df["MostFavourite_Group"] == self.df["MostFavourite"]).astype(int)
        )
        return self
        
    # -----------------------------
    # 4. Switching Behavior
    # -----------------------------
    def add_brand_switcher(self):
        self.df["Brand_Switcher"] = (
            self.df["BUMO_Group"] != self.df["BUMO_Previous_Group"]
        ).astype(int)
        return self

    # -----------------------------
    # 5. Funnel Depth (0–7)
    # -----------------------------
    def add_funnel_depth(self):
        funnel_cols = list(self.flag_map.values())
        self.df["Funnel_Depth"] = self.df[funnel_cols].sum(axis=1)
        return self

    # -----------------------------
    # 6. Awareness – Usage Gap
    # -----------------------------
    def add_awareness_usage_gap(self):
        self.df["Awareness_Usage_Gap"] = (
            self.df["Awareness_flag"] - self.df["Trial_flag"]
        )
        return self

    # -----------------------------
    # 7. Need_is_Drinking
    # -----------------------------
    def add_need_is_drinking(self):
        self.df["Need_is_Drinking"] = self.df["Needstates"].str.contains(
            "Drinking", case=False, na=False
        ).astype(int)
        return self

    # -----------------------------
    # 8. Run all steps
    # -----------------------------
    def build_all(self):
        return (self.add_brand_grouping()
                .add_brand_flags()
                .add_brand_loyalty()
                .add_brand_switcher()
                .add_funnel_depth()
                .add_awareness_usage_gap()
                .add_need_is_drinking()
                .df)

if __name__ == "__main__":
    fe = FeatureEngineering(df)
    df_fe = fe.build_all()
    print("After FeatureEngineering:", df_fe.shape)
    print(df_fe.head())
    output_path = "feature_engineering_data.csv"
    df_fe.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"File đã được lưu tại: {output_path}")