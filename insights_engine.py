#!/usr/bin/env python3
"""
Universal Insights Engine — Schema-agnostic, works with ANY MongoDB database.
Auto-discovers collections, classifies fields, generates & ranks the top 5 insights.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId
from collections import Counter
import streamlit as st

# ═══════════════════════════════════════════════════════════════
# PREMIUM CHART THEME
# ═══════════════════════════════════════════════════════════════

COLORS = {
    "primary":    "#6C63FF",
    "secondary":  "#00D9A6",
    "accent":     "#FF6B8A",
    "warning":    "#FFB347",
    "info":       "#4FC3F7",
    "success":    "#66BB6A",
    "danger":     "#EF5350",
    "muted":      "#78909C",
}

PALETTE = ["#6C63FF", "#00D9A6", "#FF6B8A", "#FFB347", "#4FC3F7",
           "#AB47BC", "#26C6DA", "#FFA726", "#EC407A", "#66BB6A"]

DARK_BG      = "rgba(0,0,0,0)"
GRID_COLOR   = "rgba(255,255,255,0.06)"
TEXT_COLOR    = "#E0E0E0"
SUBTEXT      = "#9E9E9E"


def _apply_premium_layout(fig, title="", height=420, show_legend=True):
    """Apply consistent premium dark theme to any Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=TEXT_COLOR, family="Inter, sans-serif"), x=0.02, y=0.97),
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font=dict(color=TEXT_COLOR, family="Inter, sans-serif", size=12),
        height=height, margin=dict(l=20, r=20, t=50, b=20),
        showlegend=show_legend,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.08)",
                    borderwidth=1, font=dict(size=11, color=SUBTEXT)),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, tickfont=dict(color=SUBTEXT)),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, tickfont=dict(color=SUBTEXT)),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# SCHEMA ANALYZER — Auto-discover any database
# ═══════════════════════════════════════════════════════════════

class FieldInfo:
    """Metadata about a single field in a collection."""
    def __init__(self, name, field_type, unique_count=0, sample_values=None, null_pct=0.0):
        self.name = name
        self.field_type = field_type
        self.unique_count = unique_count
        self.sample_values = sample_values or []
        self.null_pct = null_pct

    def __repr__(self):
        return f"FieldInfo({self.name}: {self.field_type}, uniq={self.unique_count})"


class CollectionSchema:
    """Schema info for a single collection."""
    def __init__(self, name, doc_count, fields):
        self.name = name
        self.doc_count = doc_count
        self.fields = fields

    @property
    def numeric_fields(self):
        return [f for f in self.fields.values() if f.field_type == "numeric"]

    @property
    def categorical_fields(self):
        return [f for f in self.fields.values() if f.field_type == "categorical"]

    @property
    def temporal_fields(self):
        return [f for f in self.fields.values() if f.field_type == "temporal"]

    @property
    def text_fields(self):
        return [f for f in self.fields.values() if f.field_type == "text"]

    @property
    def id_fields(self):
        return [f for f in self.fields.values() if f.field_type == "id"]


class SchemaAnalyzer:
    """Auto-discovers and classifies the schema of any MongoDB database."""

    SAMPLE_SIZE = 50

    def __init__(self, db):
        self.db = db

    def analyze(self):
        schemas = {}
        for coll_name in self.db.list_collection_names():
            coll = self.db[coll_name]
            doc_count = coll.estimated_document_count()
            if doc_count == 0:
                continue
            samples = list(coll.find().limit(self.SAMPLE_SIZE))
            fields = self._analyze_fields(samples, doc_count)
            schemas[coll_name] = CollectionSchema(coll_name, doc_count, fields)
        return schemas

    def _analyze_fields(self, samples, doc_count):
        field_values = {}
        n_samples = len(samples)

        for doc in samples:
            for key, val in doc.items():
                if key == "_id":
                    continue
                if key not in field_values:
                    field_values[key] = []
                field_values[key].append(val)

        fields = {}
        for fname, values in field_values.items():
            non_null = [v for v in values if v is not None]
            null_pct = (n_samples - len(non_null)) / n_samples if n_samples > 0 else 0

            if not non_null:
                fields[fname] = FieldInfo(fname, "other", null_pct=null_pct)
                continue

            unique_values = set()
            for v in non_null:
                try:
                    if isinstance(v, (list, dict)):
                        continue
                    unique_values.add(v)
                except TypeError:
                    continue

            unique_count = len(unique_values)
            sample_vals = list(unique_values)[:10]
            ftype = self._classify_field(fname, non_null, unique_count, n_samples)
            fields[fname] = FieldInfo(fname, ftype, unique_count, sample_vals, null_pct)

        return fields

    def _classify_field(self, name, values, unique_count, n_samples):
        first_val = values[0]
        if isinstance(first_val, ObjectId):
            return "id"
        if name.lower().endswith("id") or name.lower().endswith("_id"):
            return "id"
        if isinstance(first_val, datetime):
            return "temporal"
        if isinstance(first_val, bool):
            return "boolean" if unique_count <= 2 else "categorical"
        if isinstance(first_val, (int, float)):
            return "numeric"
        if isinstance(first_val, str):
            return "categorical" if unique_count <= 25 else "text"
        return "other"


# ═══════════════════════════════════════════════════════════════
# INSIGHT GENERATORS — Schema-driven, universal
# ═══════════════════════════════════════════════════════════════

class Insight:
    """A single generated insight with chart, takeaway, and metadata."""
    def __init__(self, title, fig, takeaway, metrics, score=0.0, category=""):
        self.title = title
        self.fig = fig
        self.takeaway = takeaway
        self.metrics = metrics
        self.score = score
        self.category = category


class InsightGenerators:
    """Generates insights from any collection based on field types."""

    def __init__(self, db):
        self.db = db

    def generate_all(self, schemas):
        insights = []
        for coll_name, schema in schemas.items():
            insights.extend(self._generate_for_collection(coll_name, schema))
        return insights

    def _generate_for_collection(self, coll_name, schema):
        insights = []
        coll = self.db[coll_name]

        for cat_field in schema.categorical_fields:
            ins = self._gen_categorical_distribution(coll, coll_name, cat_field, schema)
            if ins:
                insights.append(ins)

        for num_field in schema.numeric_fields:
            ins = self._gen_numeric_distribution(coll, coll_name, num_field, schema)
            if ins:
                insights.append(ins)

        for cat_field in schema.categorical_fields:
            for num_field in schema.numeric_fields:
                ins = self._gen_group_average(coll, coll_name, cat_field, num_field, schema)
                if ins:
                    insights.append(ins)

        for name_field in (schema.text_fields + schema.categorical_fields):
            for num_field in schema.numeric_fields:
                ins = self._gen_top_n(coll, coll_name, name_field, num_field, schema)
                if ins:
                    insights.append(ins)

        for time_field in schema.temporal_fields:
            for num_field in schema.numeric_fields:
                ins = self._gen_time_trend(coll, coll_name, time_field, num_field, schema)
                if ins:
                    insights.append(ins)

        num_fields = schema.numeric_fields
        for i in range(len(num_fields)):
            for j in range(i + 1, len(num_fields)):
                ins = self._gen_correlation(coll, coll_name, num_fields[i], num_fields[j], schema)
                if ins:
                    insights.append(ins)

        for time_field in schema.temporal_fields:
            ins = self._gen_time_count_trend(coll, coll_name, time_field, schema)
            if ins:
                insights.append(ins)

        return insights

    # ── Generator 1: Categorical Distribution ────────────────

    def _gen_categorical_distribution(self, coll, coll_name, field, schema):
        try:
            pipeline = [
                {"$match": {field.name: {"$exists": True, "$ne": None}}},
                {"$group": {"_id": f"${field.name}", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 15},
            ]
            results = list(coll.aggregate(pipeline))
            if len(results) < 2:
                return None

            df = pd.DataFrame(results).rename(columns={"_id": field.name})
            total = df["count"].sum()
            top = df.iloc[0]
            top_pct = top["count"] / total * 100

            title = f"{field.name.replace('_', ' ').title()} Distribution in {coll_name}"
            takeaway = f"**{top[field.name]}** is the most common {field.name} ({top_pct:.0f}%, {top['count']:,} records)."

            fig = go.Figure(go.Pie(
                labels=df[field.name], values=df["count"], hole=0.5,
                marker=dict(colors=PALETTE[:len(df)], line=dict(color=DARK_BG, width=2)),
                textinfo="percent+label", textfont=dict(size=11, color=TEXT_COLOR),
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
                pull=[0.04 if i == 0 else 0 for i in range(len(df))],
            ))
            fig.add_annotation(
                text=f"<b>{total:,}</b><br><span style='font-size:11px;color:{SUBTEXT}'>Total</span>",
                x=0.5, y=0.5, showarrow=False, font=dict(size=18, color=TEXT_COLOR),
            )
            _apply_premium_layout(fig, title, height=400)

            metrics = {"Total Records": f"{total:,}", "Categories": str(len(df)), "Top": str(top[field.name])}
            score = min(len(df), 10) / 10 * 0.6 + (1 - top_pct / 100) * 0.4
            return Insight(title, fig, takeaway, metrics, score, "distribution")
        except Exception:
            return None

    # ── Generator 2: Numeric Distribution ─────────────────────

    def _gen_numeric_distribution(self, coll, coll_name, field, schema):
        try:
            pipeline = [
                {"$match": {field.name: {"$exists": True, "$ne": None}}},
                {"$group": {"_id": None, "min": {"$min": f"${field.name}"},
                            "max": {"$max": f"${field.name}"}, "avg": {"$avg": f"${field.name}"},
                            "count": {"$sum": 1}}},
            ]
            stats = list(coll.aggregate(pipeline))
            if not stats or stats[0]["count"] < 5:
                return None
            s = stats[0]
            if s["min"] == s["max"]:
                return None

            values = [doc[field.name] for doc in coll.find(
                {field.name: {"$exists": True, "$ne": None}},
                {field.name: 1, "_id": 0}).limit(2000)]

            df = pd.DataFrame({field.name: values})
            title = f"{field.name.replace('_', ' ').title()} Distribution in {coll_name}"
            takeaway = (f"**{field.name}** ranges from **{s['min']:,.2f}** to **{s['max']:,.2f}** "
                        f"(avg: {s['avg']:,.2f}, {s['count']:,} records).")

            fig = go.Figure(go.Histogram(
                x=df[field.name], nbinsx=30,
                marker=dict(color=COLORS["primary"], line=dict(width=0), opacity=0.8),
                hovertemplate=f"{field.name}: %{{x}}<br>Count: %{{y}}<extra></extra>",
            ))
            _apply_premium_layout(fig, title, height=380, show_legend=False)
            fig.update_layout(xaxis_title=field.name.replace("_", " ").title(), yaxis_title="Count")

            metrics = {"Min": f"{s['min']:,.2f}", "Max": f"{s['max']:,.2f}",
                       "Average": f"{s['avg']:,.2f}", "Records": f"{s['count']:,}"}

            spread = (s["max"] - s["min"]) / abs(s["avg"]) if s["avg"] != 0 else 0
            score = min(spread / 5, 1.0) * 0.7 + min(s["count"] / 100, 1.0) * 0.3
            return Insight(title, fig, takeaway, metrics, score, "histogram")
        except Exception:
            return None

    # ── Generator 3: Group Average ────────────────────────────

    def _gen_group_average(self, coll, coll_name, cat_field, num_field, schema):
        try:
            pipeline = [
                {"$match": {cat_field.name: {"$exists": True, "$ne": None},
                            num_field.name: {"$exists": True, "$ne": None}}},
                {"$group": {"_id": f"${cat_field.name}",
                            "avg_value": {"$avg": f"${num_field.name}"},
                            "count": {"$sum": 1}}},
                {"$sort": {"avg_value": -1}},
                {"$limit": 15},
            ]
            results = list(coll.aggregate(pipeline))
            if len(results) < 2:
                return None

            df = pd.DataFrame(results).rename(columns={"_id": cat_field.name})
            top = df.iloc[0]
            bottom = df.iloc[-1]

            title = f"Avg {num_field.name.replace('_',' ').title()} by {cat_field.name.replace('_',' ').title()}"
            gap = ((top["avg_value"] - bottom["avg_value"]) / abs(bottom["avg_value"]) * 100) if bottom["avg_value"] != 0 else 0
            takeaway = (f"**{top[cat_field.name]}** has the highest avg {num_field.name} "
                        f"({top['avg_value']:,.2f}), **{gap:.0f}% higher** than {bottom[cat_field.name]}.")

            bar_colors = [COLORS["secondary"] if i == 0 else COLORS["primary"] for i in range(len(df))]
            fig = go.Figure(go.Bar(
                x=df[cat_field.name], y=df["avg_value"],
                marker=dict(color=bar_colors, line=dict(width=0), opacity=0.9),
                text=df["avg_value"].apply(lambda x: f"{x:,.1f}"),
                textposition="outside", textfont=dict(size=11, color=TEXT_COLOR),
            ))
            _apply_premium_layout(fig, title, height=400, show_legend=False)

            metrics = {f"Top {cat_field.name}": str(top[cat_field.name]),
                       "Highest Avg": f"{top['avg_value']:,.2f}", "Groups": str(len(df))}

            score = min(abs(gap) / 100, 1.0) * 0.5 + min(len(df), 10) / 10 * 0.3 + min(df["count"].sum() / 100, 1.0) * 0.2
            return Insight(title, fig, takeaway, metrics, score, "group_avg")
        except Exception:
            return None

    # ── Generator 4: Top N ────────────────────────────────────

    def _gen_top_n(self, coll, coll_name, name_field, num_field, schema):
        try:
            pipeline = [
                {"$match": {name_field.name: {"$exists": True, "$ne": None},
                            num_field.name: {"$exists": True, "$ne": None}}},
                {"$sort": {num_field.name: -1}},
                {"$limit": 10},
                {"$project": {name_field.name: 1, num_field.name: 1, "_id": 0}},
            ]
            results = list(coll.aggregate(pipeline))
            if len(results) < 3:
                return None

            df = pd.DataFrame(results)
            df = df.sort_values(num_field.name, ascending=True)
            top = df.iloc[-1]

            title = f"Top 10 {coll_name} by {num_field.name.replace('_',' ').title()}"
            takeaway = f"**{top[name_field.name]}** leads with {num_field.name} = **{top[num_field.name]:,.2f}**."

            fig = go.Figure(go.Bar(
                y=df[name_field.name].astype(str), x=df[num_field.name], orientation="h",
                marker=dict(color=COLORS["primary"], line=dict(width=0), opacity=0.85),
                text=df[num_field.name].apply(lambda x: f"{x:,.2f}"), textposition="outside",
                textfont=dict(size=10, color=TEXT_COLOR),
                hovertemplate="<b>%{y}</b><br>" + num_field.name + ": %{x:,.2f}<extra></extra>",
            ))
            _apply_premium_layout(fig, title, height=420, show_legend=False)
            fig.update_layout(yaxis_title="", margin=dict(l=10, r=80))

            metrics = {"Top": str(top[name_field.name]),
                       f"Best {num_field.name}": f"{top[num_field.name]:,.2f}", "Shown": f"{len(df)}"}
            score = 0.5
            return Insight(title, fig, takeaway, metrics, score, "top_n")
        except Exception:
            return None

    # ── Generator 5: Time Trend ───────────────────────────────

    def _gen_time_trend(self, coll, coll_name, time_field, num_field, schema):
        try:
            pipeline = [
                {"$match": {time_field.name: {"$exists": True, "$ne": None},
                            num_field.name: {"$exists": True, "$ne": None}}},
                {"$group": {"_id": {"year": {"$year": f"${time_field.name}"},
                                    "month": {"$month": f"${time_field.name}"}},
                            "total": {"$sum": f"${num_field.name}"},
                            "avg": {"$avg": f"${num_field.name}"},
                            "count": {"$sum": 1}}},
                {"$sort": {"_id.year": 1, "_id.month": 1}},
            ]
            results = list(coll.aggregate(pipeline))
            if len(results) < 3:
                return None

            df = pd.DataFrame(results)
            df["period"] = df["_id"].apply(lambda x: f"{x['year']}-{x['month']:02d}")
            df = df.drop(columns=["_id"])

            best = df.loc[df["total"].idxmax()]
            title = f"{num_field.name.replace('_',' ').title()} Trend Over Time ({coll_name})"
            takeaway = (f"Peak month: **{best['period']}** with total {num_field.name} = "
                        f"**{best['total']:,.2f}** across {best['count']:,} records.")

            fig = go.Figure(go.Scatter(
                x=df["period"], y=df["total"], mode="lines+markers", fill="tozeroy",
                line=dict(color=COLORS["primary"], width=3),
                marker=dict(size=6, color=COLORS["primary"]),
                fillcolor="rgba(108, 99, 255, 0.1)",
                hovertemplate="<b>%{x}</b><br>Total: %{y:,.2f}<extra></extra>",
            ))
            _apply_premium_layout(fig, title, height=400, show_legend=False)
            fig.update_layout(xaxis_title="Month", yaxis_title=num_field.name.replace("_", " ").title())

            metrics = {"Peak Month": best["period"], "Peak Total": f"{best['total']:,.2f}", "Periods": str(len(df))}
            score = 0.85 + min(len(df) / 20, 0.15)
            return Insight(title, fig, takeaway, metrics, score, "time_trend")
        except Exception:
            return None

    # ── Generator 6: Correlation ──────────────────────────────

    def _gen_correlation(self, coll, coll_name, field_a, field_b, schema):
        try:
            docs = list(coll.find(
                {field_a.name: {"$exists": True, "$ne": None},
                 field_b.name: {"$exists": True, "$ne": None}},
                {field_a.name: 1, field_b.name: 1, "_id": 0}).limit(1000))

            if len(docs) < 10:
                return None

            df = pd.DataFrame(docs)
            corr = df[field_a.name].corr(df[field_b.name])
            if abs(corr) < 0.15:
                return None

            title = f"{field_a.name.replace('_',' ').title()} vs {field_b.name.replace('_',' ').title()} ({coll_name})"
            strength = "strong" if abs(corr) > 0.6 else "moderate"
            direction = "positive" if corr > 0 else "negative"
            takeaway = (f"**{strength.title()} {direction} correlation** (r={corr:.2f}) between "
                        f"{field_a.name} and {field_b.name}.")

            fig = go.Figure(go.Scatter(
                x=df[field_a.name], y=df[field_b.name], mode="markers",
                marker=dict(color=COLORS["primary"], size=5, opacity=0.5, line=dict(width=0)),
                hovertemplate=f"{field_a.name}: %{{x:,.2f}}<br>{field_b.name}: %{{y:,.2f}}<extra></extra>",
            ))
            z = np.polyfit(df[field_a.name], df[field_b.name], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df[field_a.name].min(), df[field_a.name].max(), 100)
            fig.add_trace(go.Scatter(
                x=x_line, y=p(x_line), mode="lines",
                line=dict(color=COLORS["accent"], width=2, dash="dash"), name="Trend",
            ))
            _apply_premium_layout(fig, title, height=400)
            fig.update_layout(xaxis_title=field_a.name.replace("_", " ").title(),
                              yaxis_title=field_b.name.replace("_", " ").title())

            metrics = {"Correlation": f"{corr:.2f}", "Strength": strength.title(), "Data Points": f"{len(df):,}"}
            score = abs(corr) * 0.8 + min(len(df) / 500, 0.2)
            return Insight(title, fig, takeaway, metrics, score, "correlation")
        except Exception:
            return None

    # ── Generator 7: Time Count Trend ─────────────────────────

    def _gen_time_count_trend(self, coll, coll_name, time_field, schema):
        try:
            pipeline = [
                {"$match": {time_field.name: {"$exists": True, "$ne": None}}},
                {"$group": {"_id": {"year": {"$year": f"${time_field.name}"},
                                    "month": {"$month": f"${time_field.name}"}},
                            "count": {"$sum": 1}}},
                {"$sort": {"_id.year": 1, "_id.month": 1}},
            ]
            results = list(coll.aggregate(pipeline))
            if len(results) < 3:
                return None

            df = pd.DataFrame(results)
            df["period"] = df["_id"].apply(lambda x: f"{x['year']}-{x['month']:02d}")
            df = df.drop(columns=["_id"])

            best = df.loc[df["count"].idxmax()]
            title = f"{coll_name.replace('_',' ').title()} Volume Over Time"
            takeaway = f"Peak activity: **{best['period']}** with **{best['count']:,}** records."

            fig = go.Figure(go.Bar(
                x=df["period"], y=df["count"],
                marker=dict(color=COLORS["secondary"], line=dict(width=0), opacity=0.85),
                text=df["count"].apply(lambda x: f"{x:,}"), textposition="outside",
                textfont=dict(size=10, color=TEXT_COLOR),
                hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>",
            ))
            _apply_premium_layout(fig, title, height=380, show_legend=False)
            fig.update_layout(xaxis_title="Month", yaxis_title="Count")

            metrics = {"Peak Month": best["period"], "Peak Count": f"{best['count']:,}", "Periods": str(len(df))}
            score = 0.7
            return Insight(title, fig, takeaway, metrics, score, "count_trend")
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════
# INSIGHT RANKER — Pick the top 5 most interesting
# ═══════════════════════════════════════════════════════════════

class InsightRanker:
    @staticmethod
    def rank(insights, top_n=5):
        if not insights:
            return []

        sorted_insights = sorted(insights, key=lambda i: i.score, reverse=True)
        picked = []
        category_counts = Counter()

        for ins in sorted_insights:
            if category_counts[ins.category] >= 2:
                continue
            picked.append(ins)
            category_counts[ins.category] += 1
            if len(picked) >= top_n:
                break

        if len(picked) < top_n:
            for ins in sorted_insights:
                if ins not in picked:
                    picked.append(ins)
                    if len(picked) >= top_n:
                        break

        return picked


# ═══════════════════════════════════════════════════════════════
# UNIVERSAL INSIGHTS ENGINE — Orchestrator
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def _get_insights_db():
    client = MongoClient("mongodb://localhost:27017")
    return client["ai_test_db"]


class UniversalInsightsEngine:
    """Schema-agnostic insights engine. Works with ANY MongoDB database."""

    def __init__(self):
        self.db = _get_insights_db()
        self._schemas = None
        self._insights_cache = None

    @property
    def schemas(self):
        if self._schemas is None:
            analyzer = SchemaAnalyzer(self.db)
            self._schemas = analyzer.analyze()
        return self._schemas

    def get_top_insights(self, n=5):
        if self._insights_cache is not None:
            return self._insights_cache
        generator = InsightGenerators(self.db)
        all_insights = generator.generate_all(self.schemas)
        top = InsightRanker.rank(all_insights, top_n=n)
        self._insights_cache = top
        return top

    def get_schema_summary(self):
        lines = []
        for name, schema in self.schemas.items():
            fields_by_type = {}
            for f in schema.fields.values():
                fields_by_type.setdefault(f.field_type, []).append(f.name)
            field_desc = ", ".join(f"{k}: {', '.join(v)}" for k, v in fields_by_type.items())
            lines.append(f"**{name}** ({schema.doc_count:,} docs) — {field_desc}")
        return "\n\n".join(lines)

    def generate_custom_insight(self, question):
        question_lower = question.lower().strip()

        # Find the most relevant collection
        best_coll = None
        best_score = 0
        for name, schema in self.schemas.items():
            score = 0
            if name.lower() in question_lower:
                score += 5
            for f in schema.fields.values():
                if f.name.lower() in question_lower:
                    score += 2
                for word in question_lower.split():
                    if word in f.name.lower() or f.name.lower() in word:
                        score += 1
            if score > best_score:
                best_score = score
                best_coll = name

        if not best_coll:
            best_coll = max(self.schemas.keys(), key=lambda k: self.schemas[k].doc_count)

        schema = self.schemas[best_coll]
        coll = self.db[best_coll]
        q = question_lower

        # Distribution requests
        if any(w in q for w in ["distribution", "breakdown", "split", "proportion", "pie", "donut"]):
            for cat in schema.categorical_fields:
                if cat.name.lower() in q:
                    gen = InsightGenerators(self.db)
                    return self._wrap_insight(gen._gen_categorical_distribution(coll, best_coll, cat, schema))
            if schema.categorical_fields:
                gen = InsightGenerators(self.db)
                return self._wrap_insight(gen._gen_categorical_distribution(coll, best_coll, schema.categorical_fields[0], schema))

        # Trend requests
        if any(w in q for w in ["trend", "over time", "monthly", "timeline", "growth"]):
            if schema.temporal_fields and schema.numeric_fields:
                gen = InsightGenerators(self.db)
                return self._wrap_insight(gen._gen_time_trend(coll, best_coll, schema.temporal_fields[0], schema.numeric_fields[0], schema))
            if schema.temporal_fields:
                gen = InsightGenerators(self.db)
                return self._wrap_insight(gen._gen_time_count_trend(coll, best_coll, schema.temporal_fields[0], schema))

        # Top/best requests
        if any(w in q for w in ["top", "best", "highest", "most", "largest", "biggest"]):
            name_field = (schema.text_fields + schema.categorical_fields)
            if name_field and schema.numeric_fields:
                num_field = schema.numeric_fields[0]
                for nf in schema.numeric_fields:
                    if nf.name.lower() in q:
                        num_field = nf
                        break
                gen = InsightGenerators(self.db)
                return self._wrap_insight(gen._gen_top_n(coll, best_coll, name_field[0], num_field, schema))

        # Bottom/worst requests
        if any(w in q for w in ["bottom", "worst", "lowest", "least", "smallest"]):
            name_field = (schema.text_fields + schema.categorical_fields)
            if name_field and schema.numeric_fields:
                num_field = schema.numeric_fields[0]
                for nf in schema.numeric_fields:
                    if nf.name.lower() in q:
                        num_field = nf
                        break
                try:
                    pipeline = [
                        {"$match": {name_field[0].name: {"$exists": True}, num_field.name: {"$exists": True}}},
                        {"$sort": {num_field.name: 1}}, {"$limit": 10},
                        {"$project": {name_field[0].name: 1, num_field.name: 1, "_id": 0}},
                    ]
                    results = list(coll.aggregate(pipeline))
                    if results:
                        df = pd.DataFrame(results)
                        title = f"Bottom 10 {best_coll} by {num_field.name}"
                        takeaway = f"Lowest: **{df.iloc[0][name_field[0].name]}** with {num_field.name} = **{df.iloc[0][num_field.name]:,.2f}**."
                        fig = go.Figure(go.Bar(
                            y=df[name_field[0].name].astype(str), x=df[num_field.name],
                            orientation="h", marker=dict(color=COLORS["danger"], opacity=0.85),
                            text=df[num_field.name].apply(lambda x: f"{x:,.2f}"), textposition="outside",
                            textfont=dict(size=10, color=TEXT_COLOR),
                        ))
                        _apply_premium_layout(fig, title, height=420, show_legend=False)
                        fig.update_layout(yaxis_title="", margin=dict(l=10, r=80))
                        return fig, takeaway, df
                except Exception:
                    pass

        # Average/compare requests
        if any(w in q for w in ["average", "avg", "compare", "comparison", "by"]):
            if schema.categorical_fields and schema.numeric_fields:
                cat = schema.categorical_fields[0]
                num = schema.numeric_fields[0]
                for cf in schema.categorical_fields:
                    if cf.name.lower() in q:
                        cat = cf
                        break
                for nf in schema.numeric_fields:
                    if nf.name.lower() in q:
                        num = nf
                        break
                gen = InsightGenerators(self.db)
                return self._wrap_insight(gen._gen_group_average(coll, best_coll, cat, num, schema))

        # Correlation requests
        if any(w in q for w in ["correlation", "relationship", "vs", "versus", "scatter"]):
            if len(schema.numeric_fields) >= 2:
                fa, fb = schema.numeric_fields[0], schema.numeric_fields[1]
                for nf in schema.numeric_fields:
                    if nf.name.lower() in q and nf != fa:
                        fb = nf
                gen = InsightGenerators(self.db)
                return self._wrap_insight(gen._gen_correlation(coll, best_coll, fa, fb, schema))

        # Fallback: best auto-generated insight
        gen = InsightGenerators(self.db)
        all_ins = gen._generate_for_collection(best_coll, schema)
        if all_ins:
            best = max(all_ins, key=lambda i: i.score)
            return best.fig, best.takeaway, None

        available = []
        for name, s in self.schemas.items():
            fields = [f.name for f in list(s.fields.values())[:5]]
            available.append(f"  - **{name}**: {', '.join(fields)}")
        avail_text = "\n".join(available)
        return None, f"I couldn't generate an insight for that question. Available collections:\n\n{avail_text}", None

    def _wrap_insight(self, insight):
        if insight is None:
            return None, "Not enough data for this insight.", None
        return insight.fig, insight.takeaway, None
