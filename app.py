from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from shapely.geometry import LineString, Point, Polygon
from streamlit_folium import st_folium

try:
    import scipy.optimize as opt

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
QATAR_CRS = "EPSG:4326"
QATAR_METRIC_CRS = "EPSG:2933"
ELECTRICITY_QAR_KWH = 0.20
WATER_QAR_M3 = 5.0
COOLING_HOURS_YEAR = 1900
T_SET_C = 25.0


QATAR_POLYGON = Polygon(
    [
        (50.755, 24.595),
        (50.925, 24.570),
        (51.130, 24.590),
        (51.355, 24.705),
        (51.520, 24.905),
        (51.595, 25.090),
        (51.635, 25.300),
        (51.590, 25.545),
        (51.515, 25.760),
        (51.405, 25.965),
        (51.245, 26.135),
        (51.065, 26.155),
        (50.925, 25.980),
        (50.805, 25.720),
        (50.755, 25.390),
        (50.745, 25.020),
        (50.755, 24.595),
    ]
)


REFERENCE_STATIONS = {
    "Doha": (51.531, 25.285, 41.5, 55.0, 890.0),
    "Al Khor": (51.507, 25.684, 39.8, 62.0, 870.0),
    "Al Shahaniya": (51.205, 25.371, 43.2, 48.0, 920.0),
    "Mesaieed": (51.553, 24.998, 42.8, 52.0, 905.0),
    "Ruwais": (51.221, 26.142, 39.2, 65.0, 860.0),
}


@dataclass(frozen=True)
class CropProfile:
    name: str
    crop_family: str
    yield_kg_m2_year: float
    kc: float
    price_qar_kg: float
    water_sensitivity: float
    climate_sensitivity: float
    preferred_zone: str
    notes: str


@dataclass(frozen=True)
class GreenhouseTech:
    name: str
    category: str
    capital_qar_m2: float
    fixed_opex_qar_m2_year: float
    cooling_mode: str
    cop: float
    pad_efficiency: float
    lighting_w_m2: float
    yield_multiplier: float
    water_multiplier: float
    energy_multiplier: float
    labour_factor: float
    description: str


class AdvancedGreenhouseEngine:
    CP_AIR = 1006.0
    RHO_AIR = 1.204
    LAMBDA_LATENT = 2.45e6
    G_ACCEL = 9.81

    def __init__(self, specs: dict):
        self.length = specs["length"]
        self.width = specs["width"]
        self.height_eaves = specs["height_eaves"]
        self.height_roof = specs["height_roof"]
        self.area_floor = self.length * self.width
        self.volume = self.area_floor * ((self.height_eaves + self.height_roof) / 2.0)
        self.area_cover = specs["area_cover"]
        self.area_vent_roof = specs["area_vent_roof"]
        self.area_vent_side = specs["area_vent_side"]
        self.vent_height_diff = self.height_roof - self.height_eaves
        self.tau_cover = specs["tau_cover"]
        self.net_porosity = specs["net_porosity"]
        self.lai = specs["lai"]
        self.rc_min = specs["rc_min"]

    @staticmethod
    def saturation_vapor_pressure(temp_c: float) -> float:
        return 0.61078 * np.exp((17.27 * temp_c) / (temp_c + 237.3))

    def ventilation_rate(self, temp_in: float, temp_out: float, wind_m_s: float) -> float:
        cd_net = 0.6 * (self.net_porosity**2)
        area_eff = (self.area_vent_roof * self.area_vent_side) / math.sqrt(self.area_vent_roof**2 + self.area_vent_side**2 + 1e-6)
        wind_flow = cd_net * area_eff * max(wind_m_s, 0.1) * math.sqrt(0.22)
        temp_avg_k = (temp_in + temp_out + 273.15) / 2.0
        delta_t = max(abs(temp_in - temp_out), 0.001)
        buoyancy_flow = cd_net * self.area_vent_roof * math.sqrt((2.0 * self.G_ACCEL * self.vent_height_diff * delta_t) / temp_avg_k)
        return math.sqrt(wind_flow**2 + buoyancy_flow**2)

    def evaporative_pad_outlet(self, temp_out: float, rh_out: float, pad_efficiency: float) -> tuple[float, float]:
        # Stull-style wet-bulb approximation keeps the model stable for dashboard use.
        twb = (
            temp_out * math.atan(0.151977 * math.sqrt(rh_out + 8.313659))
            + math.atan(temp_out + rh_out)
            - math.atan(rh_out - 1.676331)
            + 0.00391838 * rh_out**1.5 * math.atan(0.023101 * rh_out)
            - 4.686035
        )
        temp_pad = temp_out - pad_efficiency * (temp_out - twb)
        rh_pad = min(100.0, rh_out + pad_efficiency * (100.0 - rh_out))
        return temp_pad, rh_pad

    def equilibrium_state(self, boundary: dict) -> dict:
        solar = boundary["solar_w_m2"]
        temp_out = boundary["temp_c"]
        rh_out = boundary["rh_pct"]
        wind = boundary.get("wind_m_s", 3.0)
        pad_active = boundary.get("pad_active", False)
        pad_eff = boundary.get("pad_efficiency", 0.84)
        shading = boundary.get("shading_factor", 0.35)

        temp_inlet, rh_inlet = self.evaporative_pad_outlet(temp_out, rh_out, pad_eff) if pad_active else (temp_out, rh_out)
        solar_net = solar * self.tau_cover * (1.0 - shading) * self.area_floor

        def residual(states):
            temp_in, rh_in = float(states[0]), float(np.clip(states[1], 5.0, 100.0))
            vent = self.ventilation_rate(temp_in, temp_out, wind)
            conduction = 5.8 * self.area_cover * (temp_in - temp_out)
            sensible_vent = vent * self.RHO_AIR * self.CP_AIR * (temp_in - temp_inlet)
            p_sat = self.saturation_vapor_pressure(temp_in)
            vapor_pressure = p_sat * rh_in / 100.0
            vpd = max(0.0, p_sat - vapor_pressure)
            aerodynamic_resistance = 200.0 / math.sqrt(max(wind, 0.1))
            stomatal_resistance = self.rc_min * (1.0 + 100.0 / max(1.0, solar * self.tau_cover))
            transpiration_w = (self.area_floor * self.lai * self.RHO_AIR * self.CP_AIR * (vpd / 0.066)) / (aerodynamic_resistance + stomatal_resistance)
            sensible = solar_net * 0.45 - conduction - sensible_vent - transpiration_w * 0.1
            w_inlet = 0.622 * (self.saturation_vapor_pressure(temp_inlet) * rh_inlet / 100.0) / 101.3
            w_internal = 0.622 * vapor_pressure / 101.3
            latent = transpiration_w / self.LAMBDA_LATENT - vent * self.RHO_AIR * (w_internal - w_inlet)
            return [sensible, latent]

        if SCIPY_AVAILABLE:
            solution = opt.root(residual, [temp_out + 4.0, min(95.0, rh_out + 12.0)], method="hybr")
            temp_in = float(solution.x[0])
            rh_in = float(np.clip(solution.x[1], 5.0, 100.0))
            solver_success = bool(solution.success)
        else:
            temp_in = temp_inlet + max(1.0, solar_net / max(self.area_floor, 1.0) / 125.0)
            rh_in = min(100.0, rh_inlet + 6.0)
            solver_success = False

        vent_final = self.ventilation_rate(temp_in, temp_out, wind)
        return {
            "internal_temperature_c": temp_in,
            "internal_relative_humidity_pct": rh_in,
            "ventilation_rate_m3_s": vent_final,
            "air_changes_per_hour": (vent_final * 3600.0) / max(self.volume, 1.0),
            "solver_success": solver_success,
        }


CROP_DATABASE: Dict[str, CropProfile] = {
    "Tomato - truss/cherry": CropProfile("Tomato", "fruiting vegetable", 34.0, 1.10, 8.0, 0.72, 0.75, "Inland low humidity", "High value, high cooling sensitivity"),
    "Cucumber - long": CropProfile("Cucumber", "fruiting vegetable", 46.0, 1.00, 5.8, 0.55, 0.58, "Inland central plains", "Fast cycles and strong ventilation demand"),
    "Sweet pepper": CropProfile("Sweet pepper", "fruiting vegetable", 24.0, 1.05, 12.0, 0.65, 0.82, "Low humidity inland", "Sensitive to heat stress and flower drop"),
    "Lettuce": CropProfile("Lettuce", "leafy green", 28.0, 0.78, 7.0, 0.38, 0.42, "Coastal controlled systems", "Short cycle, suitable for stacked production"),
    "Strawberry": CropProfile("Strawberry", "berry", 14.0, 0.90, 24.0, 0.78, 0.88, "Fully controlled/coastal", "High value but climate sensitive"),
    "Eggplant": CropProfile("Eggplant", "fruiting vegetable", 31.0, 0.98, 6.5, 0.58, 0.62, "Inland farms", "Robust crop with moderate value"),
    "Melon - netted": CropProfile("Melon", "vine crop", 18.0, 0.95, 9.5, 0.68, 0.68, "Inland large bays", "Needs space and careful humidity management"),
    "Basil and herbs": CropProfile("Basil and herbs", "herb", 18.5, 0.72, 18.0, 0.35, 0.45, "Controlled or semi-controlled", "High price, compact production"),
}


GREENHOUSE_TECHS: Dict[str, GreenhouseTech] = {
    "Low-tech shade net": GreenhouseTech(
        "Low-tech shade net",
        "low-tech",
        120,
        18,
        "passive",
        0.0,
        0.0,
        0.0,
        0.58,
        0.92,
        0.45,
        1.15,
        "Lowest capital cost, weak summer climate control; best only for tolerant crops and seasonal windows.",
    ),
    "Fan-pad evaporative": GreenhouseTech(
        "Fan-pad evaporative",
        "mid-tech",
        360,
        42,
        "evaporative",
        0.0,
        0.84,
        0.0,
        1.00,
        1.18,
        1.00,
        1.00,
        "Efficient inland where relative humidity is lower; high cooling water demand.",
    ),
    "Hybrid pad + chiller": GreenhouseTech(
        "Hybrid pad + chiller",
        "high-tech",
        1050,
        95,
        "hybrid",
        2.7,
        0.78,
        0.0,
        1.18,
        0.68,
        1.55,
        0.90,
        "Balanced system that shifts from evaporative cooling to mechanical cooling in humid periods.",
    ),
    "Mechanical chiller": GreenhouseTech(
        "Mechanical chiller",
        "high-tech",
        880,
        82,
        "mechanical",
        3.2,
        0.0,
        0.0,
        1.10,
        0.18,
        1.85,
        0.86,
        "Minimizes cooling water use but raises electrical demand and grid dependency.",
    ),
    "Fully controlled + LED": GreenhouseTech(
        "Fully controlled + LED",
        "CEA",
        1800,
        155,
        "mechanical",
        3.6,
        0.0,
        32.0,
        1.36,
        0.12,
        2.20,
        0.72,
        "Highest control, highest capital intensity; suited to premium crops and compact production.",
    ),
}


LAND_USE_COLORS = {
    "Agricultural": "#2f855a",
    "Open desert/rangeland": "#d6a84f",
    "Unclassified open land": "#c2a95f",
    "Residential": "#d43d3d",
    "Industrial": "#6b7280",
    "Protected area": "#7c3aed",
    "Flood-prone depression": "#f97316",
    "Urban expansion buffer": "#e11d48",
    "Water/offshore": "#2563eb",
}


def synthetic_lines(name: str) -> gpd.GeoDataFrame:
    if name == "power":
        geometries = [
            LineString([(51.03, 24.88), (51.14, 25.05), (51.28, 25.29), (51.47, 25.43)]),
            LineString([(50.93, 25.36), (51.17, 25.31), (51.42, 25.28)]),
            LineString([(51.12, 25.67), (51.25, 25.45), (51.35, 25.31)]),
            LineString([(51.00, 24.74), (51.22, 24.93), (51.55, 25.02)]),
        ]
    else:
        geometries = [
            LineString([(51.18, 24.72), (51.25, 25.05), (51.32, 25.35), (51.45, 25.71), (51.51, 26.02)]),
            LineString([(50.88, 25.22), (51.08, 25.25), (51.27, 25.31), (51.53, 25.37)]),
            LineString([(50.92, 25.10), (51.12, 25.05), (51.35, 25.04)]),
            LineString([(50.88, 25.60), (51.05, 25.54), (51.24, 25.45)]),
        ]
    return gpd.GeoDataFrame(
        {"name": [f"synthetic_{name}_{i + 1}" for i in range(len(geometries))]},
        geometry=geometries,
        crs=QATAR_CRS,
    )


def synthetic_landuse() -> gpd.GeoDataFrame:
    records = [
        ("Al Shahaniya agriculture", "Agricultural", True, Polygon([(50.88, 25.12), (51.20, 25.10), (51.22, 25.38), (50.90, 25.39)])),
        ("Rawdat Rashed agriculture", "Agricultural", True, Polygon([(50.93, 24.85), (51.22, 24.82), (51.24, 25.05), (50.96, 25.08)])),
        ("Umm Salal agriculture", "Agricultural", True, Polygon([(51.24, 25.38), (51.48, 25.37), (51.46, 25.61), (51.22, 25.60)])),
        ("Northern open land", "Open desert/rangeland", True, Polygon([(50.92, 25.62), (51.22, 25.58), (51.30, 25.92), (51.05, 26.05), (50.86, 25.86)])),
        ("Southern open land", "Open desert/rangeland", True, Polygon([(50.84, 24.66), (51.10, 24.62), (51.18, 24.82), (50.92, 24.95)])),
        ("Doha residential/urban", "Residential", False, Point(51.53, 25.29).buffer(0.105)),
        ("Al Wakrah residential", "Residential", False, Point(51.60, 25.17).buffer(0.060)),
        ("Al Khor residential", "Residential", False, Point(51.51, 25.68).buffer(0.060)),
        ("Umm Salal residential", "Residential", False, Point(51.40, 25.42).buffer(0.045)),
        ("Mesaieed industrial", "Industrial", False, Point(51.55, 24.99).buffer(0.070)),
        ("Dukhan industrial", "Industrial", False, Point(50.79, 25.42).buffer(0.060)),
        ("Al Reem protected area", "Protected area", False, Point(50.88, 25.67).buffer(0.115)),
        ("Rawda flood-prone depression", "Flood-prone depression", False, Point(51.10, 25.15).buffer(0.075)),
        ("Doha urban expansion buffer", "Urban expansion buffer", False, Point(51.42, 25.30).buffer(0.085)),
    ]
    return gpd.GeoDataFrame(
        {
            "name": [record[0] for record in records],
            "landuse": [record[1] for record in records],
            "greenhouse_ok": [record[2] for record in records],
        },
        geometry=[record[3] for record in records],
        crs=QATAR_CRS,
    )


def off_land_result() -> dict:
    return {
        "feasible": False,
        "landuse": "Water/offshore",
        "landuse_name": "Outside the Qatar land mask",
        "crop": "Not applicable",
        "technology": "Not applicable",
        "temp_c": np.nan,
        "rh_pct": np.nan,
        "ghi_w_m2": np.nan,
        "et0_mm_day": np.nan,
        "yield_tons": 0.0,
        "irrigation_m3": 0.0,
        "cooling_water_m3": 0.0,
        "total_water_m3": 0.0,
        "total_energy_mwh": 0.0,
        "peak_cooling_kw": 0.0,
        "water_l_kg": np.nan,
        "energy_kwh_kg": np.nan,
        "internal_temperature_c": np.nan,
        "internal_relative_humidity_pct": np.nan,
        "ventilation_rate_m3_s": 0.0,
        "air_changes_per_hour": 0.0,
        "microclimate_solver": False,
        "capital_qar": 0.0,
        "opex_qar": 0.0,
        "revenue_qar": 0.0,
        "net_profit_qar": 0.0,
        "payback_years": float("inf"),
        "roi_percent": 0.0,
    }


@st.cache_data(show_spinner=False)
def load_vector_layer(filename: str, fallback: str) -> gpd.GeoDataFrame:
    path = DATA_DIR / filename
    if path.exists():
        gdf = gpd.read_file(path).to_crs(QATAR_CRS)
        if fallback == "landuse":
            if "landuse" not in gdf.columns:
                gdf["landuse"] = gdf.get("class", "Unknown")
            if "greenhouse_ok" not in gdf.columns:
                allowed = {"agricultural", "agriculture", "farm", "open desert", "rangeland", "bare land"}
                gdf["greenhouse_ok"] = gdf["landuse"].astype(str).str.lower().isin(allowed)
        return gdf
    if fallback == "power":
        return synthetic_lines("power")
    if fallback == "roads":
        return synthetic_lines("roads")
    return synthetic_landuse()


def allowed_landuse(landuse: str) -> bool:
    return landuse in {"Agricultural", "Open desert/rangeland", "Unclassified open land"}


def split_landuse(landuse: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    allowed = landuse[landuse["greenhouse_ok"].astype(bool)].copy()
    excluded = landuse[~landuse["greenhouse_ok"].astype(bool)].copy()
    return allowed, excluded


def point_distance_m(point: Point, layer: gpd.GeoDataFrame) -> float:
    if layer.empty:
        return float("inf")
    point_gdf = gpd.GeoDataFrame(geometry=[point], crs=QATAR_CRS).to_crs(QATAR_METRIC_CRS)
    projected = layer.to_crs(QATAR_METRIC_CRS)
    return float(point_gdf.geometry.iloc[0].distance(projected.geometry).min())


def polygon_distance_m(poly: Polygon, layer: gpd.GeoDataFrame) -> float:
    if layer.empty:
        return float("inf")
    poly_gdf = gpd.GeoDataFrame(geometry=[poly], crs=QATAR_CRS).to_crs(QATAR_METRIC_CRS)
    projected = layer.to_crs(QATAR_METRIC_CRS)
    return float(poly_gdf.geometry.iloc[0].distance(projected.geometry).min())


def point_landuse(point: Point, landuse: gpd.GeoDataFrame) -> dict:
    hits = landuse[landuse.intersects(point)]
    if hits.empty:
        return {"landuse": "Unclassified open land", "greenhouse_ok": True, "name": "Requires official land-use verification"}
    disallowed = hits[~hits["greenhouse_ok"].astype(bool)]
    selected = disallowed.iloc[0] if not disallowed.empty else hits.iloc[0]
    return {
        "landuse": str(selected.get("landuse", "Unknown")),
        "greenhouse_ok": bool(selected.get("greenhouse_ok", False)),
        "name": str(selected.get("name", "Land-use polygon")),
    }


def normalize_distance(distance_m: float, ideal_m: float, max_m: float) -> float:
    if distance_m <= ideal_m:
        return 1.0
    if distance_m >= max_m:
        return 0.0
    return 1.0 - ((distance_m - ideal_m) / (max_m - ideal_m))


def interpolate_climate(lat: float, lon: float) -> dict:
    point = Point(lon, lat)
    if not QATAR_POLYGON.contains(point):
        return {"temp_c": np.nan, "rh_pct": np.nan, "ghi_w_m2": np.nan, "station_note": "water/offshore"}

    weighted = []
    for name, (station_lon, station_lat, temp_c, rh_pct, ghi_w_m2) in REFERENCE_STATIONS.items():
        distance = max(math.hypot(lon - station_lon, lat - station_lat), 0.0001)
        weight = 1.0 / (distance**2)
        weighted.append((weight, name, temp_c, rh_pct, ghi_w_m2))

    weight_sum = sum(row[0] for row in weighted)
    temp = sum(weight * temp for weight, _, temp, _, _ in weighted) / weight_sum
    rh = sum(weight * rh for weight, _, _, rh, _ in weighted) / weight_sum
    ghi = sum(weight * ghi for weight, _, _, _, ghi in weighted) / weight_sum
    nearest = min(weighted, key=lambda row: 1.0 / math.sqrt(row[0]))[1]
    return {"temp_c": round(temp, 1), "rh_pct": round(rh, 1), "ghi_w_m2": round(ghi, 0), "station_note": f"IDW; nearest {nearest}"}


def et0_hargreaves_mm_day(temp_c: float, ghi_w_m2: float) -> float:
    if not np.isfinite(temp_c) or not np.isfinite(ghi_w_m2):
        return 0.0
    solar_mj_m2_day = ghi_w_m2 * 12.0 * 3600.0 / 1_000_000.0
    et0 = 0.0023 * solar_mj_m2_day * math.sqrt(12.0) * (temp_c + 17.8)
    return round(max(1.5, et0), 2)


def cooling_load_kw(area_m2: float, transmissivity: float, climate: dict, tech: GreenhouseTech) -> float:
    if not np.isfinite(climate["temp_c"]) or not np.isfinite(climate["ghi_w_m2"]):
        return 0.0
    solar_gain_w = area_m2 * climate["ghi_w_m2"] * transmissivity * 0.88
    delta_t = max(0.0, climate["temp_c"] - T_SET_C)
    sensible_w = 1.4 * (area_m2 * 0.55) * delta_t
    humidity_penalty = max(0.0, climate["rh_pct"] - 55.0) / 100.0
    latent_w = (solar_gain_w + sensible_w) * humidity_penalty * (0.55 if tech.cooling_mode in {"evaporative", "hybrid"} else 0.25)
    return (solar_gain_w + sensible_w + latent_w) / 1000.0


def greenhouse_geometry(area_m2: int, transmissivity: float, crop: CropProfile) -> dict:
    bay_width = 9.6
    length = max(24.0, area_m2 / bay_width)
    cover_factor = 1.78
    return {
        "length": length,
        "width": bay_width,
        "height_eaves": 4.0,
        "height_roof": 5.6,
        "area_cover": area_m2 * cover_factor,
        "area_vent_roof": area_m2 * 0.065,
        "area_vent_side": area_m2 * 0.095,
        "tau_cover": transmissivity,
        "net_porosity": 0.52,
        "lai": 2.6 if crop.crop_family == "fruiting vegetable" else 1.8,
        "rc_min": 120.0 if crop.crop_family == "fruiting vegetable" else 95.0,
    }


def microclimate_design_state(area_m2: int, transmissivity: float, crop: CropProfile, tech: GreenhouseTech, climate: dict) -> dict:
    if not np.isfinite(climate["temp_c"]):
        return {
            "internal_temperature_c": np.nan,
            "internal_relative_humidity_pct": np.nan,
            "ventilation_rate_m3_s": 0.0,
            "air_changes_per_hour": 0.0,
            "solver_success": False,
        }
    engine = AdvancedGreenhouseEngine(greenhouse_geometry(area_m2, transmissivity, crop))
    pad_active = tech.cooling_mode in {"evaporative", "hybrid"}
    shading = 0.45 if tech.cooling_mode == "passive" else 0.35
    if tech.name == "Fully controlled + LED":
        shading = 0.28
    return engine.equilibrium_state(
        {
            "solar_w_m2": climate["ghi_w_m2"],
            "temp_c": climate["temp_c"],
            "rh_pct": climate["rh_pct"],
            "wind_m_s": 3.2,
            "pad_active": pad_active,
            "pad_efficiency": max(tech.pad_efficiency, 0.84) if pad_active else 0.0,
            "shading_factor": shading,
        }
    )


def analyze_location(lat: float, lon: float, crop: CropProfile, tech: GreenhouseTech, area_m2: int, transmissivity: float, recycle_drainage: bool, landuse: gpd.GeoDataFrame) -> dict:
    point = Point(lon, lat)
    if not QATAR_POLYGON.contains(point):
        result = off_land_result()
        result["crop"] = crop.name
        result["technology"] = tech.name
        return result
    lu = point_landuse(point, landuse)
    climate = interpolate_climate(lat, lon)
    et0 = et0_hargreaves_mm_day(climate["temp_c"], climate["ghi_w_m2"])
    irrigation_m3 = (et0 * crop.kc / 1000.0) * area_m2 * 365.0
    if recycle_drainage:
        irrigation_m3 *= 0.70

    peak_kw = cooling_load_kw(area_m2, transmissivity, climate, tech)
    rh_penalty = 1.0 + max(0.0, (climate["rh_pct"] - 55.0) / 55.0)

    if tech.cooling_mode == "passive":
        cooling_water_m3 = 0.04 * peak_kw * COOLING_HOURS_YEAR / 1000.0
        cooling_energy_mwh = area_m2 * 12.0 / 1000.0
    elif tech.cooling_mode == "evaporative":
        cooling_water_m3 = peak_kw * COOLING_HOURS_YEAR * 0.00115 * rh_penalty / max(tech.pad_efficiency, 0.1)
        fan_kw = area_m2 * 0.0042
        pump_kw = area_m2 * 0.0010
        cooling_energy_mwh = (fan_kw + pump_kw) * COOLING_HOURS_YEAR / 1000.0
    elif tech.cooling_mode == "hybrid":
        evap_fraction = max(0.20, min(0.70, 1.0 - (climate["rh_pct"] - 45.0) / 45.0))
        evap_water = peak_kw * evap_fraction * COOLING_HOURS_YEAR * 0.00095 * rh_penalty / max(tech.pad_efficiency, 0.1)
        chiller_energy = peak_kw * (1.0 - evap_fraction) * COOLING_HOURS_YEAR / max(tech.cop, 0.1) / 1000.0
        fan_energy = area_m2 * 0.0035 * COOLING_HOURS_YEAR / 1000.0
        cooling_water_m3 = evap_water
        cooling_energy_mwh = chiller_energy + fan_energy
    else:
        cooling_water_m3 = peak_kw * COOLING_HOURS_YEAR * 0.00008
        cooling_energy_mwh = peak_kw * COOLING_HOURS_YEAR / max(tech.cop, 0.1) / 1000.0

    base_energy_mwh = area_m2 * 34.0 / 1000.0
    lighting_mwh = tech.lighting_w_m2 * area_m2 * 2000.0 / 1_000_000.0
    total_energy_mwh = (cooling_energy_mwh + base_energy_mwh + lighting_mwh) * tech.energy_multiplier
    total_water_m3 = (irrigation_m3 + cooling_water_m3) * tech.water_multiplier
    micro_state = microclimate_design_state(area_m2, transmissivity, crop, tech, climate)

    stress = 1.0
    if climate["temp_c"] > 40.0 and tech.cooling_mode in {"passive", "evaporative"}:
        stress -= crop.climate_sensitivity * 0.16
    if climate["rh_pct"] > 60.0 and tech.cooling_mode == "evaporative":
        stress -= crop.climate_sensitivity * 0.12
    if not lu["greenhouse_ok"]:
        stress = 0.0
    stress = max(0.0, stress)

    yield_kg = crop.yield_kg_m2_year * area_m2 * tech.yield_multiplier * stress
    revenue_qar = yield_kg * crop.price_qar_kg
    capital_qar = area_m2 * tech.capital_qar_m2
    fixed_opex_qar = area_m2 * tech.fixed_opex_qar_m2_year
    electricity_qar = total_energy_mwh * 1000.0 * ELECTRICITY_QAR_KWH
    water_qar = total_water_m3 * WATER_QAR_M3
    labour_qar = (area_m2 / 1000.0) * 0.12 * 5000.0 * 12.0 * tech.labour_factor
    maintenance_qar = capital_qar * 0.035
    total_opex_qar = fixed_opex_qar + electricity_qar + water_qar + labour_qar + maintenance_qar
    net_profit_qar = revenue_qar - total_opex_qar
    payback_years = capital_qar / net_profit_qar if net_profit_qar > 0 else float("inf")
    roi_percent = net_profit_qar / capital_qar * 100.0 if capital_qar > 0 else 0.0

    return {
        "feasible": bool(lu["greenhouse_ok"]),
        "landuse": lu["landuse"],
        "landuse_name": lu["name"],
        "crop": crop.name,
        "technology": tech.name,
        "temp_c": climate["temp_c"],
        "rh_pct": climate["rh_pct"],
        "ghi_w_m2": climate["ghi_w_m2"],
        "et0_mm_day": et0,
        "yield_tons": yield_kg / 1000.0,
        "irrigation_m3": irrigation_m3,
        "cooling_water_m3": cooling_water_m3,
        "total_water_m3": total_water_m3,
        "total_energy_mwh": total_energy_mwh,
        "peak_cooling_kw": peak_kw,
        "water_l_kg": (total_water_m3 * 1000.0 / yield_kg) if yield_kg > 0 else float("inf"),
        "energy_kwh_kg": (total_energy_mwh * 1000.0 / yield_kg) if yield_kg > 0 else float("inf"),
        "internal_temperature_c": micro_state["internal_temperature_c"],
        "internal_relative_humidity_pct": micro_state["internal_relative_humidity_pct"],
        "ventilation_rate_m3_s": micro_state["ventilation_rate_m3_s"],
        "air_changes_per_hour": micro_state["air_changes_per_hour"],
        "microclimate_solver": micro_state["solver_success"],
        "capital_qar": capital_qar,
        "opex_qar": total_opex_qar,
        "revenue_qar": revenue_qar,
        "net_profit_qar": net_profit_qar,
        "payback_years": payback_years,
        "roi_percent": roi_percent,
    }


def calculate_suitability(lat: float, lon: float, weights: dict, layers: dict) -> dict:
    point = Point(lon, lat)
    if not QATAR_POLYGON.contains(point):
        return {
            "score": 0.0,
            "status": "Water/offshore: outside the Qatar land mask",
            "is_excluded": True,
            "landuse": "Water/offshore",
            "landuse_name": "No greenhouse analysis is calculated for water",
            "landuse_score": 0.0,
            "grid_distance_m": float("inf"),
            "road_distance_m": float("inf"),
            "excluded_distance_m": float("inf"),
            "climate_score": 0.0,
            "grid_score": 0.0,
            "road_score": 0.0,
            "constraint_score": 0.0,
        }

    climate = interpolate_climate(lat, lon)
    lu = point_landuse(point, layers["landuse"])
    grid_distance = point_distance_m(point, layers["power"])
    road_distance = point_distance_m(point, layers["roads"])
    excluded_distance = point_distance_m(point, layers["excluded_landuse"])

    climate_score = max(0.0, min(1.0, 1.0 - (climate["rh_pct"] - 38.0) / 35.0))
    grid_score = normalize_distance(grid_distance, 800.0, 28_000.0)
    road_score = normalize_distance(road_distance, 1_200.0, 24_000.0)
    constraint_score = min(1.0, excluded_distance / 1500.0)
    if lu["greenhouse_ok"]:
        if lu["landuse"] == "Agricultural":
            landuse_score = 1.0
        elif lu["landuse"] == "Unclassified open land":
            landuse_score = 0.58
        else:
            landuse_score = 0.72
    else:
        landuse_score = 0.0

    raw_score = (
        climate_score * weights["climate"]
        + grid_score * weights["grid"]
        + road_score * weights["logistics"]
        + landuse_score * weights["landuse"]
        + constraint_score * weights["constraints"]
    )
    score = round(max(0.0, min(raw_score * 100.0, 100.0)), 1)
    is_excluded = not lu["greenhouse_ok"]
    if is_excluded:
        score = 0.0
        status = f"Excluded land use: {lu['landuse']}"
    elif score >= 78:
        status = "Excellent: realistic greenhouse candidate"
    elif score >= 58:
        status = "Good: viable after site verification"
    elif score >= 38:
        status = "Moderate: resource or infrastructure burden"
    else:
        status = "Low suitability"

    return {
        "score": score,
        "status": status,
        "is_excluded": is_excluded,
        "landuse": lu["landuse"],
        "landuse_name": lu["name"],
        "grid_distance_m": grid_distance,
        "road_distance_m": road_distance,
        "excluded_distance_m": excluded_distance,
        "climate_score": round(climate_score * 100.0, 1),
        "grid_score": round(grid_score * 100.0, 1),
        "road_score": round(road_score * 100.0, 1),
        "landuse_score": round(landuse_score * 100.0, 1),
        "constraint_score": round(constraint_score * 100.0, 1),
    }


def score_color(score: float, excluded: bool = False) -> str:
    if excluded:
        return "#991b1b"
    if score >= 78:
        return "#16803c"
    if score >= 58:
        return "#73a827"
    if score >= 38:
        return "#d18c00"
    return "#b33430"


def build_heatmap_runtime(weights: dict, layers: dict) -> gpd.GeoDataFrame:
    records = []
    for lat in np.linspace(24.68, 26.02, 24):
        for lon in np.linspace(50.84, 51.55, 24):
            point = Point(lon, lat)
            if not QATAR_POLYGON.contains(point):
                continue
            result = calculate_suitability(lat, lon, weights, layers)
            records.append(
                {
                    "score": result["score"],
                    "status": result["status"],
                    "landuse": result["landuse"],
                    "is_excluded": result["is_excluded"],
                    "geometry": point.buffer(0.018),
                }
            )
    return gpd.GeoDataFrame(records, crs=QATAR_CRS)


def add_geojson(map_object: folium.Map, gdf: gpd.GeoDataFrame, name: str, color: str, fill_color: Optional[str] = None, weight: int = 2, fill_opacity: float = 0.16) -> None:
    if gdf.empty:
        return
    folium.GeoJson(
        gdf,
        name=name,
        tooltip=folium.GeoJsonTooltip(fields=[col for col in ["name", "landuse", "greenhouse_ok"] if col in gdf.columns]),
        style_function=lambda _feature: {
            "color": color,
            "weight": weight,
            "fillColor": fill_color or color,
            "fillOpacity": fill_opacity,
        },
    ).add_to(map_object)


def build_map(lat: float, lon: float, weights: dict, layers: dict, show_heatmap: bool, show_infra: bool, show_landuse: bool) -> folium.Map:
    map_object = folium.Map(location=[25.3548, 51.1839], zoom_start=9, tiles="CartoDB positron", control_scale=True)
    folium.Rectangle(
        bounds=[[24.48, 50.66], [26.22, 51.72]],
        color="#60a5fa",
        weight=0,
        fill=True,
        fill_color="#dbeafe",
        fill_opacity=0.22,
        tooltip="Water/offshore areas are excluded from greenhouse analysis",
    ).add_to(map_object)
    boundary = gpd.GeoDataFrame({"name": ["Qatar analysis boundary"]}, geometry=[QATAR_POLYGON], crs=QATAR_CRS)
    add_geojson(map_object, boundary, "Qatar land mask", "#374151", fill_color="#f9fafb", fill_opacity=0.42)

    if show_heatmap:
        heatmap = build_heatmap_runtime(weights, layers)
        folium.GeoJson(
            heatmap,
            name="Feasible suitability heatmap",
            tooltip=folium.GeoJsonTooltip(fields=["score", "status", "landuse"]),
            style_function=lambda feature: {
                "fillColor": score_color(float(feature["properties"]["score"]), bool(feature["properties"]["is_excluded"])),
                "color": score_color(float(feature["properties"]["score"]), bool(feature["properties"]["is_excluded"])),
                "weight": 0.35,
                "fillOpacity": 0.46 if not feature["properties"]["is_excluded"] else 0.62,
            },
        ).add_to(map_object)

    if show_landuse:
        for landuse_name, color in LAND_USE_COLORS.items():
            subset = layers["landuse"][layers["landuse"]["landuse"] == landuse_name]
            opacity = 0.18 if allowed_landuse(landuse_name) else 0.34
            add_geojson(map_object, subset, f"Land use: {landuse_name}", color, fill_color=color, weight=2, fill_opacity=opacity)

    if show_infra:
        add_geojson(map_object, layers["power"], "Power grid proxy", "#c0262d", weight=4)
        add_geojson(map_object, layers["roads"], "Primary highways", "#2563eb", weight=4)

    marker_color = "red" if not QATAR_POLYGON.contains(Point(lon, lat)) else "green"
    marker_text = "Selected point - water/offshore" if marker_color == "red" else "Selected greenhouse site"
    folium.Marker([lat, lon], tooltip=marker_text, icon=folium.Icon(color=marker_color, icon="leaf")).add_to(map_object)
    folium.LayerControl(collapsed=True).add_to(map_object)
    map_object.fit_bounds([[24.55, 50.72], [26.18, 51.66]])
    return map_object


def format_distance(distance_m: float) -> str:
    if math.isinf(distance_m):
        return "Unavailable"
    if distance_m >= 1000:
        return f"{distance_m / 1000:,.1f} km"
    return f"{distance_m:,.0f} m"


def money(value: float) -> str:
    return f"QAR {value:,.0f}"


def years(value: float) -> str:
    if math.isinf(value) or value > 100:
        return ">100 years"
    return f"{value:.1f} years"


def all_combinations(lat: float, lon: float, area_m2: int, transmissivity: float, recycle: bool, landuse: gpd.GeoDataFrame) -> pd.DataFrame:
    rows = []
    for crop in CROP_DATABASE.values():
        for tech in GREENHOUSE_TECHS.values():
            rows.append(analyze_location(lat, lon, crop, tech, area_m2, transmissivity, recycle, landuse))
    return pd.DataFrame(rows)


def build_pdf_report(site: dict, report_df: pd.DataFrame) -> Optional[bytes]:
    if not REPORTLAB_AVAILABLE:
        return None
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("Qatar Greenhouse Atlas - Site Report", styles["Title"]),
        Paragraph(f"Coordinates: {site['lat']:.4f} N, {site['lon']:.4f} E", styles["Normal"]),
        Paragraph(f"Land use: {site['landuse']} | Suitability: {site['score']}/100", styles["Normal"]),
        Spacer(1, 12),
    ]
    table_df = report_df.head(10)[["crop", "technology", "yield_tons", "total_water_m3", "total_energy_mwh", "net_profit_qar", "payback_years"]].copy()
    table_df.columns = ["Crop", "Technology", "Yield t", "Water m3", "Energy MWh", "Profit QAR", "Payback"]
    data = [list(table_df.columns)] + table_df.round(2).astype(str).values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9ca3af")),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    elements.append(table)
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("Planning-grade model. Replace synthetic land-use and infrastructure layers with official GeoJSON layers for regulatory use.", styles["Italic"]))
    doc.build(elements)
    return buffer.getvalue()


st.set_page_config(page_title="Qatar Greenhouse Atlas", page_icon="🌱", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 0.65rem; padding-bottom: 1.4rem; max-width: 1500px;}
    div[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 0.58rem;}
    div[data-testid="stMetric"] label {font-size: 0.78rem;}
    div[data-testid="stMetricValue"] {font-size: 1.05rem;}
    .small-note {font-size: 0.86rem; color: #4b5563;}
    </style>
    """,
    unsafe_allow_html=True,
)

power_lines = load_vector_layer("qatar_kahramaa_lines.geojson", "power")
roads = load_vector_layer("qatar_ashghal_roads.geojson", "roads")
landuse = load_vector_layer("qatar_landuse.geojson", "landuse")
allowed_landuse_gdf, excluded_landuse_gdf = split_landuse(landuse)
layers = {
    "power": power_lines,
    "roads": roads,
    "landuse": landuse,
    "allowed_landuse": allowed_landuse_gdf,
    "excluded_landuse": excluded_landuse_gdf,
}

if "selected_lat" not in st.session_state:
    st.session_state.selected_lat = 25.3548
    st.session_state.selected_lon = 51.1839

st.title("Qatar Greenhouse Atlas")
st.caption("Land, water, land-use, crop-technology optimisation, and greenhouse investment screening.")

with st.sidebar:
    st.header("Suitability Weights")
    w_climate = st.slider("Low-humidity microclimate", 0.0, 1.0, 0.26, 0.01)
    w_grid = st.slider("Power grid proximity", 0.0, 1.0, 0.18, 0.01)
    w_logistics = st.slider("Highway logistics", 0.0, 1.0, 0.14, 0.01)
    w_landuse = st.slider("Permitted land use", 0.0, 1.0, 0.32, 0.01)
    w_constraints = st.slider("Buffer from exclusions", 0.0, 1.0, 0.10, 0.01)
    total_weight = max(w_climate + w_grid + w_logistics + w_landuse + w_constraints, 0.01)
    weights = {
        "climate": w_climate / total_weight,
        "grid": w_grid / total_weight,
        "logistics": w_logistics / total_weight,
        "landuse": w_landuse / total_weight,
        "constraints": w_constraints / total_weight,
    }

    st.header("Production Assumptions")
    area_m2 = st.number_input("Greenhouse area (m²)", min_value=500, max_value=250_000, value=5_000, step=500)
    transmissivity = st.slider("Cover transmissivity", 0.45, 0.85, 0.65, 0.01)
    recycle = st.toggle("Closed-loop hydroponic drainage", value=True)

    st.header("Map Layers")
    show_heatmap = st.toggle("Suitability heatmap", value=True)
    show_landuse = st.toggle("Land-use layer", value=True)
    show_infra = st.toggle("Infrastructure layers", value=True)

tab_map, tab_compare, tab_opt, tab_notes = st.tabs(["Suitability Map", "Crop-Tech Comparison", "Investment & Optimisation", "Model Notes"])

lat = float(st.session_state.selected_lat)
lon = float(st.session_state.selected_lon)
suitability = calculate_suitability(lat, lon, weights, layers)
climate = interpolate_climate(lat, lon)
report_df = all_combinations(lat, lon, int(area_m2), transmissivity, recycle, landuse)

with tab_map:
    map_col, metric_col = st.columns([2.15, 0.85], gap="medium")
    with map_col:
        st.subheader("National Feasibility Map")
        map_object = build_map(lat, lon, weights, layers, show_heatmap, show_infra, show_landuse)
        map_data = st_folium(map_object, height=760, use_container_width=True)
        if map_data and map_data.get("last_clicked"):
            st.session_state.selected_lat = map_data["last_clicked"]["lat"]
            st.session_state.selected_lon = map_data["last_clicked"]["lng"]
            st.rerun()

    with metric_col:
        st.subheader("Selected Site")
        if suitability["is_excluded"]:
            st.error(f"{suitability['status']}. No greenhouse production or investment analysis is valid here.")
        else:
            st.success(suitability["status"])
        st.metric("Suitability index", f"{suitability['score']}/100")
        st.metric("Coordinates", f"{lat:.4f} N, {lon:.4f} E")
        st.metric("Land use", suitability["landuse"], suitability.get("landuse_name", ""))
        if np.isfinite(climate["temp_c"]):
            st.metric("Summer climate", f"{climate['temp_c']} °C / {climate['rh_pct']}% RH", f"{climate['ghi_w_m2']:.0f} W/m² GHI")
        else:
            st.metric("Summer climate", "Not calculated", "water/offshore")
        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Grid distance", format_distance(float(suitability["grid_distance_m"])))
        c2.metric("Road distance", format_distance(float(suitability["road_distance_m"])))
        c1.metric("Land-use score", f"{suitability['landuse_score']:.0f}/100")
        c2.metric("Exclusion buffer", format_distance(float(suitability["excluded_distance_m"])))

        default_report = analyze_location(lat, lon, CROP_DATABASE["Tomato - truss/cherry"], GREENHOUSE_TECHS["Fan-pad evaporative"], int(area_m2), transmissivity, recycle, landuse)
        with st.expander("Site report: tomato + fan-pad", expanded=not suitability["is_excluded"]):
            r1, r2 = st.columns(2)
            r1.metric("Annual yield", f"{default_report['yield_tons']:,.1f} t")
            r2.metric("Total water", f"{default_report['total_water_m3']:,.0f} m³/yr")
            r1.metric("Total energy", f"{default_report['total_energy_mwh']:,.0f} MWh/yr")
            r2.metric("Peak cooling", f"{default_report['peak_cooling_kw']:,.0f} kW")
            r1.metric("Indoor temp", "N/A" if not np.isfinite(default_report["internal_temperature_c"]) else f"{default_report['internal_temperature_c']:.1f} °C")
            r2.metric("ACH", f"{default_report['air_changes_per_hour']:.0f}/h")
            st.metric("Capital investment", money(default_report["capital_qar"]))
            st.metric("Net annual profit", money(default_report["net_profit_qar"]))
            st.metric("Payback", years(default_report["payback_years"]), f"ROI {default_report['roi_percent']:.1f}%")

    score_df = pd.DataFrame(
        {
            "Factor": ["Microclimate", "Power grid", "Highway logistics", "Permitted land use", "Exclusion buffer"],
            "Score": [
                suitability["climate_score"],
                suitability["grid_score"],
                suitability["road_score"],
                suitability["landuse_score"],
                suitability["constraint_score"],
            ],
        }
    )
    st.subheader("Suitability Score Breakdown")
    st.bar_chart(score_df, x="Factor", y="Score", height=270)

with tab_compare:
    st.subheader("Compare Crop and Greenhouse Technology Packages")
    pair_cols = st.columns(3)
    selected_reports = []
    for idx, col in enumerate(pair_cols):
        with col:
            crop_name = st.selectbox(f"Crop {idx + 1}", list(CROP_DATABASE.keys()), index=min(idx, len(CROP_DATABASE) - 1), key=f"crop_{idx}")
            tech_name = st.selectbox(f"Technology {idx + 1}", list(GREENHOUSE_TECHS.keys()), index=min(idx + 1, len(GREENHOUSE_TECHS) - 1), key=f"tech_{idx}")
            selected_reports.append(analyze_location(lat, lon, CROP_DATABASE[crop_name], GREENHOUSE_TECHS[tech_name], int(area_m2), transmissivity, recycle, landuse))

    comparison_df = pd.DataFrame(selected_reports)
    display_cols = [
        "feasible",
        "landuse",
        "crop",
        "technology",
        "yield_tons",
        "total_water_m3",
        "total_energy_mwh",
        "internal_temperature_c",
        "air_changes_per_hour",
        "capital_qar",
        "net_profit_qar",
        "payback_years",
        "roi_percent",
    ]
    st.dataframe(comparison_df[display_cols], hide_index=True, width="stretch")
    st.download_button(
        "Download comparison CSV",
        comparison_df.to_csv(index=False).encode("utf-8"),
        "greenhouse_crop_technology_comparison.csv",
        "text/csv",
    )

with tab_opt:
    st.subheader("Optimisation for Current Location")
    objective = st.selectbox("Optimisation objective", ["Maximise net profit", "Minimise water use", "Minimise energy use", "Fastest payback", "Highest ROI"])
    feasible_only = st.toggle("Show feasible land-use sites only", value=True)
    opt_df = report_df.copy()
    if feasible_only:
        opt_df = opt_df[opt_df["feasible"]]

    if opt_df.empty:
        st.error("This selected point is on excluded land use. Choose an agricultural or open-land location to run feasible optimisation.")
    else:
        if objective == "Maximise net profit":
            opt_df = opt_df.sort_values("net_profit_qar", ascending=False)
        elif objective == "Minimise water use":
            opt_df = opt_df.sort_values("total_water_m3", ascending=True)
        elif objective == "Minimise energy use":
            opt_df = opt_df.sort_values("total_energy_mwh", ascending=True)
        elif objective == "Fastest payback":
            opt_df = opt_df.sort_values("payback_years", ascending=True)
        else:
            opt_df = opt_df.sort_values("roi_percent", ascending=False)

        top = opt_df.head(8)
        st.dataframe(
            top[["crop", "technology", "yield_tons", "total_water_m3", "total_energy_mwh", "internal_temperature_c", "air_changes_per_hour", "capital_qar", "net_profit_qar", "payback_years", "roi_percent"]],
            hide_index=True,
            width="stretch",
        )
        best = top.iloc[0]
        st.success(f"Best ranked option: {best['crop']} using {best['technology']} | Net profit {money(best['net_profit_qar'])} | Payback {years(best['payback_years'])}")

    site_payload = {"lat": lat, "lon": lon, "landuse": suitability["landuse"], "score": suitability["score"]}
    st.download_button("Download full optimisation CSV", report_df.to_csv(index=False).encode("utf-8"), "qatar_greenhouse_full_optimisation.csv", "text/csv")
    pdf_bytes = build_pdf_report(site_payload, report_df.sort_values("net_profit_qar", ascending=False))
    if pdf_bytes:
        st.download_button("Download PDF site report", pdf_bytes, "qatar_greenhouse_site_report.pdf", "application/pdf")
    else:
        st.info("PDF export needs the optional reportlab package. CSV export is available now.")

with tab_notes:
    st.subheader("Scientific and GIS Notes")
    st.markdown(
        f"""
        **Land-use rule:** residential, industrial, protected, flood-prone, and urban expansion polygons are hard exclusions.
        Suitability is forced to 0 on excluded land, even if climate or infrastructure scores are strong.

        **Water rule:** clicks outside the Qatar land mask are classified as `Water/offshore`. The model returns no yield,
        no water demand, no energy demand, and no investment ranking for these points.

        **Replaceable GIS layers:** add official files to `{DATA_DIR}`:
        `qatar_landuse.geojson`, `qatar_kahramaa_lines.geojson`, and `qatar_ashghal_roads.geojson`.
        A land-use file should include `landuse` and preferably `greenhouse_ok` fields.

        **Climate model:** summer temperature, relative humidity, and GHI are interpolated with inverse-distance weighting
        from five Qatar reference stations. The values are planning-grade defaults and should be replaced with measured
        gridded climate data for engineering design.

        **Water model:** irrigation is estimated from a FAO-56 style crop coefficient approach using simplified Hargreaves ET0.

        **Cooling and microclimate model:** peak cooling combines solar gain, sensible heat, and a humidity penalty.
        A deterministic greenhouse microclimate engine estimates indoor equilibrium temperature, RH, ventilation rate,
        and air changes per hour using ventilation, evaporative pad, and crop transpiration balances. Technology packages
        then translate load into cooling water, electrical demand, capital cost, OPEX, revenue, payback, and ROI.

        **Important:** this is a screening atlas. It is not a substitute for official zoning approval, utility connection
        studies, parcel ownership checks, soil/geotechnical assessment, or detailed HVAC design.
        """
    )
