import streamlit as st
import pandas as pd
import os
import json
import numpy as np
from datetime import datetime, time, timedelta
from dateutil import parser
import io
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
import hashlib
from functools import lru_cache, partial
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy.signal import find_peaks
from itertools import groupby
import shutil
import glob
import time as time_module
from websocket import create_connection, WebSocket
import random
from typing import Dict, List, Tuple, Optional, Union
import plotly.graph_objects as plotly_go
from plotly.subplots import make_subplots
import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth
import urllib3
import logging
import sqlite3
import uuid

# Disable insecure request warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Performance configuration
CACHE_VERSION = "1.0"
CACHE_DIR = ".vitals_cache"
CHUNK_SIZE = 5000  # Increased chunk size for better batch performance
MAX_WORKERS = max(4, multiprocessing.cpu_count())  # Optimize worker count
COMPRESSION = 'snappy'  # Fast compression for parquet files
DB_PATH = "vitals_data.db"  # Path to SQLite database

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

###############################################################################
# DATABASE FUNCTIONALITY
###############################################################################

def init_database():
    """Initialize the database with required tables if they don't exist"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create metadata table to track datasets
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS datasets (
        id TEXT PRIMARY KEY,
        name TEXT,
        upload_date TIMESTAMP,
        num_records INTEGER,
        description TEXT,
        source TEXT
    )
    ''')
    
    # Create table for vital signs data
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS vital_signs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id TEXT,
        TimeStr TEXT,
        TimeObj TIMESTAMP,
        DevSerial TEXT,
        SourceFile TEXT,
        rawMannequin TEXT,
        Course TEXT,
        Sim TEXT,
        overrideMannequin TEXT,
        DateStr TEXT,
        Scenario TEXT,
        InSimWindow INTEGER,
        IsValid INTEGER,
        FOREIGN KEY (dataset_id) REFERENCES datasets(id)
    )
    ''')
    
    # Create table for vital sign values
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS vital_values (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        vital_sign_id INTEGER,
        vital_name TEXT,
        vital_value REAL,
        vital_status TEXT,
        FOREIGN KEY (vital_sign_id) REFERENCES vital_signs(id)
    )
    ''')
    
    conn.commit()
    conn.close()

def save_to_database(df, name="Uploaded Data", description="", source=""):
    """
    Save DataFrame to SQLite database
    
    Args:
        df: DataFrame with vital signs data
        name: Name of the dataset
        description: Description of the dataset
        source: Source of the dataset (e.g., file path)
        
    Returns:
        dataset_id: ID of the saved dataset
    """
    if df.empty:
        return None
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Generate a unique ID for this dataset
    dataset_id = str(uuid.uuid4())
    
    # Insert dataset metadata
    cursor.execute(
        "INSERT INTO datasets (id, name, upload_date, num_records, description, source) VALUES (?, ?, ?, ?, ?, ?)",
        (dataset_id, name, datetime.now(), len(df), description, source)
    )
    
    # Insert vital signs data in batches
    batch_size = 1000
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        
        for _, row in batch_df.iterrows():
            # Insert basic vital sign record
            cursor.execute(
                """
                INSERT INTO vital_signs 
                (dataset_id, TimeStr, TimeObj, DevSerial, SourceFile, rawMannequin, 
                Course, Sim, overrideMannequin, DateStr, Scenario, InSimWindow, IsValid)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    row.get("TimeStr", None),
                    row.get("TimeObj", None).isoformat() if row.get("TimeObj") else None,
                    row.get("DevSerial", None),
                    row.get("SourceFile", None),
                    row.get("rawMannequin", None),
                    row.get("Course", None),
                    row.get("Sim", None),
                    row.get("overrideMannequin", None),
                    row.get("DateStr", None),
                    row.get("Scenario", None),
                    1 if row.get("InSimWindow", False) else 0,
                    1 if row.get("IsValid", False) else 0
                )
            )
            
            # Get the ID of the inserted vital sign record
            vital_sign_id = cursor.lastrowid
            
            # Insert vital values
            for col in row.index:
                # Skip non-vital columns and status columns
                if col in ["TimeStr", "TimeObj", "DevSerial", "SourceFile", "rawMannequin",
                          "Course", "Sim", "overrideMannequin", "DateStr", "Scenario",
                          "InSimWindow", "IsValid", "ElapsedMin"] or col.endswith("_status"):
                    continue
                
                # Get value and status
                value = row.get(col)
                status = row.get(f"{col}_status", "valid")
                
                if pd.notna(value):
                    cursor.execute(
                        "INSERT INTO vital_values (vital_sign_id, vital_name, vital_value, vital_status) VALUES (?, ?, ?, ?)",
                        (vital_sign_id, col, float(value), status)
                    )
    
    conn.commit()
    conn.close()
    
    return dataset_id

def load_from_database(dataset_id=None):
    """
    Load vital signs data from SQLite database
    
    Args:
        dataset_id: ID of the dataset to load. If None, loads the most recent dataset.
        
    Returns:
        DataFrame with vital signs data
    """
    conn = sqlite3.connect(DB_PATH)
    
    # Get dataset ID if not provided
    if dataset_id is None:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM datasets ORDER BY upload_date DESC LIMIT 1")
        result = cursor.fetchone()
        if result:
            dataset_id = result[0]
        else:
            conn.close()
            return pd.DataFrame()  # No datasets available
    
    # Load basic vital sign records
    vital_signs_df = pd.read_sql_query(
        "SELECT * FROM vital_signs WHERE dataset_id = ?",
        conn,
        params=(dataset_id,)
    )
    
    if vital_signs_df.empty:
        conn.close()
        return pd.DataFrame()
    
    # Convert boolean columns
    vital_signs_df["InSimWindow"] = vital_signs_df["InSimWindow"].astype(bool)
    vital_signs_df["IsValid"] = vital_signs_df["IsValid"].astype(bool)
    
    # Convert TimeObj back to datetime
    vital_signs_df["TimeObj"] = pd.to_datetime(vital_signs_df["TimeObj"])
    
    # Load vital values
    vital_values_df = pd.read_sql_query(
        """
        SELECT v.vital_sign_id, v.vital_name, v.vital_value, v.vital_status
        FROM vital_values v
        JOIN vital_signs s ON v.vital_sign_id = s.id
        WHERE s.dataset_id = ?
        """,
        conn,
        params=(dataset_id,)
    )
    
    conn.close()
    
    if vital_values_df.empty:
        return vital_signs_df
    
    # Pivot vital values to get them in the right format
    # First, create DataFrames for values and statuses
    values_df = vital_values_df.pivot(
        index="vital_sign_id",
        columns="vital_name",
        values="vital_value"
    )
    
    status_df = vital_values_df.pivot(
        index="vital_sign_id",
        columns="vital_name",
        values="vital_status"
    )
    
    # Rename status columns
    status_df.columns = [f"{col}_status" for col in status_df.columns]
    
    # Merge pivoted data with vital signs
    values_df.reset_index(inplace=True)
    status_df.reset_index(inplace=True)
    
    result_df = pd.merge(
        vital_signs_df,
        values_df,
        left_on="id",
        right_on="vital_sign_id",
        how="left"
    )
    
    result_df = pd.merge(
        result_df,
        status_df,
        left_on="id",
        right_on="vital_sign_id",
        how="left"
    )
    
    # Drop database IDs from the final DataFrame
    result_df.drop(columns=["id", "vital_sign_id_x", "vital_sign_id_y", "dataset_id"], 
                  errors="ignore", inplace=True)
    
    return result_df

def get_database_datasets():
    """
    Get list of available datasets in the database
    
    Returns:
        DataFrame with dataset information
    """
    conn = sqlite3.connect(DB_PATH)
    datasets_df = pd.read_sql_query(
        "SELECT id, name, upload_date, num_records, description, source FROM datasets ORDER BY upload_date DESC",
        conn
    )
    conn.close()
    
    if not datasets_df.empty:
        # Convert upload_date to datetime
        datasets_df["upload_date"] = pd.to_datetime(datasets_df["upload_date"])
        # Format the date for display
        datasets_df["upload_date_formatted"] = datasets_df["upload_date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    return datasets_df

def delete_database_dataset(dataset_id):
    """
    Delete a dataset from the database
    
    Args:
        dataset_id: ID of the dataset to delete
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # First delete related vital values
    cursor.execute(
        """
        DELETE FROM vital_values 
        WHERE vital_sign_id IN (
            SELECT id FROM vital_signs WHERE dataset_id = ?
        )
        """,
        (dataset_id,)
    )
    
    # Delete vital signs
    cursor.execute(
        "DELETE FROM vital_signs WHERE dataset_id = ?",
        (dataset_id,)
    )
    
    # Delete dataset
    cursor.execute(
        "DELETE FROM datasets WHERE id = ?",
        (dataset_id,)
    )
    
    conn.commit()
    conn.close()

###############################################################################
# EVENT DETECTION & TIME SERIES ANALYSIS
###############################################################################

def detect_vital_events(df, vital, window_size=5):
    """
    Detect significant events in vital sign trends using rolling statistics
    
    Parameters:
    - df: DataFrame with TimeObj and vital sign columns
    - vital: Name of vital sign column to analyze
    - window_size: Size of rolling window in minutes
    
    Returns:
    - List of detected events with timestamps and descriptions
    """
    events = []
    
    # Convert to numeric and sort by time
    df = df.sort_values("TimeObj").copy()
    values = pd.to_numeric(df[vital], errors='coerce')
    
    # Calculate rolling statistics
    rolling_mean = values.rolling(window=window_size, min_periods=1).mean()
    rolling_std = values.rolling(window=window_size, min_periods=1).std()
    
    # Calculate rate of change (per minute)
    df['delta_min'] = df['TimeObj'].diff().dt.total_seconds() / 60
    rate_of_change = values.diff() / df['delta_min']
    
    # Detect rapid changes (more than 2 std devs from mean change)
    mean_change = rate_of_change.mean()
    std_change = rate_of_change.std()
    
    rapid_changes = (abs(rate_of_change) > (mean_change + 2 * std_change))
    
    for idx in df[rapid_changes].index:
        events.append({
            'timestamp': df.loc[idx, 'TimeObj'],
            'type': 'rapid_change',
            'vital': vital,
            'value': values[idx],
            'rate': rate_of_change[idx],
            'description': f"Rapid change in {vital}: {rate_of_change[idx]:.1f} units/min"
        })
    
    return events

def analyze_trends(df, vital, window_size):
    """Analyze trends in vital signs data"""
    series = pd.to_numeric(df[vital], errors='coerce')
    
    # Calculate basic statistics
    mean = series.mean()
    std = series.std()
    
    # Calculate rolling statistics
    rolling_mean = series.rolling(window=window_size, min_periods=1).mean()
    rolling_std = series.rolling(window=window_size, min_periods=1).std()
    
    # Calculate rate of change (per minute)
    rate_of_change = series.diff() / df['ElapsedMin'].diff()
    
    # Determine trend
    start_window = series.iloc[:window_size].mean()
    end_window = series.iloc[-window_size:].mean()
    
    if abs(end_window - start_window) < std * 0.5:
        trend = "stable"
    elif end_window > start_window:
        trend = "increasing"
    else:
        trend = "decreasing"
    
    # Calculate stability score (0-1, lower is more stable)
    stability = rolling_std.mean() / mean if mean != 0 else 1
    
    # Generate alerts based on analysis
    alerts = []
    
    # Check for rapid changes
    rapid_change_threshold = std * 2
    if abs(rate_of_change).max() > rapid_change_threshold:
        alerts.append("Rapid changes detected")
    
    # Check for sustained deviation from mean
    if (abs(rolling_mean - mean) > std * 2).any():
        alerts.append("Sustained deviation from baseline")
    
    # Check for high variability
    if stability > 0.2:
        alerts.append("High variability")
    
    return {
        'mean': mean,
        'trend': trend,
        'stability': stability,
        'alerts': alerts
    }

def detect_clinical_events(df, scenario):
    """Detect clinically significant events in vital signs data"""
    events = []
    
    # Get scenario-specific thresholds
    thresholds = get_scenario_thresholds(scenario)
    
    # Check each vital sign
    for vital, limits in thresholds.items():
        if vital not in df.columns:
            continue
            
        series = pd.to_numeric(df[vital], errors='coerce')
        
        # Check for threshold breaches
        if 'low_critical' in limits:
            mask = series < limits['low_critical']
            if mask.any():
                events.extend([{
                    'timestamp': df['TimeObj'].iloc[i],
                    'vital': vital,
                    'type': 'critical_low',
                    'description': f"{vital} critically low: {series.iloc[i]:.1f}"
                } for i in mask[mask].index])
        
        if 'high_critical' in limits:
            mask = series > limits['high_critical']
            if mask.any():
                events.extend([{
                    'timestamp': df['TimeObj'].iloc[i],
                    'vital': vital,
                    'type': 'critical_high',
                    'description': f"{vital} critically high: {series.iloc[i]:.1f}"
                } for i in mask[mask].index])
        
        # Detect rapid changes
        rate_of_change = series.diff() / df['ElapsedMin'].diff()
        rapid_change_threshold = series.std() * 2
        
        rapid_changes = abs(rate_of_change) > rapid_change_threshold
        if rapid_changes.any():
            events.extend([{
                'timestamp': df['TimeObj'].iloc[i],
                'vital': vital,
                'type': 'rapid_change',
                'description': f"Rapid change in {vital}: {rate_of_change.iloc[i]:.1f}/min"
            } for i in rapid_changes[rapid_changes].index])
    
    # Sort events by timestamp
    events.sort(key=lambda x: x['timestamp'])
    return events

def get_scenario_thresholds(scenario):
    """Get vital sign thresholds based on scenario"""
    # Default thresholds
    thresholds = {
        'Hr': {
            'low_critical': 40,
            'high_critical': 150
        },
        'SpO2': {
            'low_critical': 90
        },
        'NIBP_MAP': {
            'low_critical': 60,
            'high_critical': 120
        },
        'Temp1': {
            'low_critical': 35,
            'high_critical': 39
        }
    }
    
    # Scenario-specific adjustments
    if 'sepsis' in scenario.lower():
        thresholds['Hr']['high_critical'] = 130
        thresholds['NIBP_MAP']['low_critical'] = 65
        thresholds['Temp1']['high_critical'] = 38.3
    elif 'shock' in scenario.lower():
        thresholds['Hr']['high_critical'] = 140
        thresholds['NIBP_MAP']['low_critical'] = 55
    elif 'respiratory' in scenario.lower():
        thresholds['SpO2']['low_critical'] = 92
    
    return thresholds

@lru_cache(maxsize=1024)  # Increased cache size
def calculate_file_hash(file_path):
    """Calculate SHA-256 hash of a file using buffered reading"""
    sha256_hash = hashlib.sha256()
    buffer_size = 65536  # 64kb chunks
    with open(file_path, "rb") as f:
        while True:
            data = f.read(buffer_size)
            if not data:
                break
            sha256_hash.update(data)
    return sha256_hash.hexdigest()

def get_cache_key(folder_path):
    """Generate a cache key based on folder contents and their hashes"""
    json_files = sorted(Path(folder_path).glob("*.json"))
    if not json_files:
        return None
    
    # Use ThreadPoolExecutor for I/O-bound hash calculation
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        hash_futures = {executor.submit(calculate_file_hash, str(f)): f.name 
                       for f in json_files}
        hash_components = [f"{hash_futures[future]}:{future.result()}" 
                         for future in hash_futures]
    
    combined = f"{CACHE_VERSION}:" + "|".join(sorted(hash_components))
    return hashlib.sha256(combined.encode()).hexdigest()

def save_to_cache(df, cache_key):
    """Save DataFrame to parquet cache with optimized settings"""
    if not cache_key:
        return
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.parquet")
    df.to_parquet(cache_path, compression=COMPRESSION, index=False)

def load_from_cache(cache_key):
    """Load DataFrame from parquet cache with optimized settings"""
    if not cache_key:
        return None
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.parquet")
    if os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path, engine='pyarrow')
        except:
            return None
    return None

@lru_cache(maxsize=2048)
def get_trend_val(data_state=None, data_status=None, val_text=None):
    """
    Enhanced version of trend value extraction that explicitly handles different states and formats
    Returns: (value, status)
    - value: The numeric value or None
    - status: 'valid', 'unmonitored', 'invalid', or 'error'
    """
    # First determine the status
    if data_state == "---" or data_state == "unmonitored" or data_state is None:
        return None, "unmonitored"
    
    # Handle various invalid states more comprehensively
    if (data_state == "invalid" or data_state == "INVALID" or 
        (isinstance(data_status, (int, str)) and str(data_status) == "1")):
        return None, "invalid"
    
    # Try to parse the value, including explicit 0
    try:
        if val_text is None:
            return None, "error"
            
        # Handle different data formats
        if isinstance(val_text, (int, float)):
            value = float(val_text)
        elif isinstance(val_text, str):
            # Remove any non-numeric characters except decimal point and negative sign
            clean_val = ''.join(c for c in val_text if c.isdigit() or c in '.-')
            if not clean_val:
                return None, "error"
            value = float(clean_val)
        elif isinstance(val_text, dict) and "#text" in val_text:
            # Handle nested dictionary format
            text_value = val_text["#text"]
            if isinstance(text_value, (int, float)):
                value = float(text_value)
            elif isinstance(text_value, str):
                clean_val = ''.join(c for c in text_value if c.isdigit() or c in '.-')
                if not clean_val:
                    return None, "error"
                value = float(clean_val)
            else:
                return None, "error"
        else:
            return None, "error"
            
        # Additional validation for blood pressure values
        # Typical ranges: SYS (70-200), DIA (40-120), MAP (50-150)
        return value, "valid"
    except (ValueError, TypeError):
        return None, "error"

def extract_trend_data(trend_data):
    """Helper function to safely extract trend data components with improved handling of nested structures"""
    if not trend_data:
        return None, None, None
        
    # Extract DataState with fallback handling
    data_state = trend_data.get("DataState")
    
    # Extract DataStatus with fallback handling
    data_status = trend_data.get("DataStatus")
    
    # Extract Val with improved handling of nested structures
    val = trend_data.get("Val")
    val_text = None
    
    if val is not None:
        if isinstance(val, dict):
            # Handle nested dictionary with #text key
            val_text = val.get("#text")
            
            # Some JSON structures have a nested 'value' key
            if val_text is None and "value" in val:
                val_text = val.get("value")
                
            # Some JSON structures have a nested 'Value' key
            if val_text is None and "Value" in val:
                val_text = val.get("Value")
        elif isinstance(val, (int, float, str)):
            val_text = val
    
    # If we have a value but no data state, assume it's valid
    if val_text is not None and data_state is None:
        data_state = "valid"
    
    return data_state, data_status, val_text

def parse_trend_rpt(tr, devS):
    """Original trend report parsing"""
    dt_s = tr.get("StdHdr",{}).get("DevDateTime")
    try:
        dt_o = parser.parse(dt_s)
    except:
        return []

    row = {
        "TimeObj": dt_o,
        "TimeStr": dt_s,
        "DevSerial": devS
    }

    trend = tr.get("Trend",{})
    
    # Process temperature readings with decimal correction
    tmpA = trend.get("Temp", [])
    for i, tA in enumerate(tmpA, start=1):
        trend_data = tA.get("TrendData", {})
        ds, st, vt = extract_trend_data(trend_data)
        temp_val, temp_status = get_trend_val(ds, st, vt)
        # Always divide temperature by 10 to correct decimal placement
        if temp_val is not None:
            try:
                temp_val = float(temp_val) / 10.0  # Always divide by 10
            except (ValueError, TypeError):
                temp_val = None
                temp_status = "error"
        row[f"Temp{i}"] = temp_val
        row[f"Temp{i}_status"] = temp_status

    # Process vital signs
    vital_mappings = {
        "Hr": ("Hr", "TrendData"),
        "FiCO2": ("Fico2", "TrendData"),
        "SpO2": ("Spo2", "TrendData"),
        "EtCO2": ("Etco2", "TrendData"),
        "RespRate": ("Resp", "TrendData")
    }

    for col_name, (trend_key, data_key) in vital_mappings.items():
        trend_data = trend.get(trend_key, {}).get(data_key, {})
        ds, st, vt = extract_trend_data(trend_data)
        val, status = get_trend_val(ds, st, vt)
        row[col_name] = val
        row[f"{col_name}_status"] = status

    # Process SpO2-related parameters
    sO2 = trend.get("Spo2",{})
    spo2_params = ["SpMet", "SpCo", "PVI", "PI", "SpOC", "SpHb"]
    for param in spo2_params:
        trend_data = sO2.get(param, {}).get("TrendData", {})
        ds, st, vt = extract_trend_data(trend_data)
        val, status = get_trend_val(ds, st, vt)
        row[param] = val
        row[f"{param}_status"] = status

    # Process NIBP readings
    nA = trend.get("Nibp",{})
    if nA:
        for component in ["Sys", "Dia", "Map"]:
            trend_data = nA.get(component.lower(), {}).get("TrendData", {})
            ds, st, vt = extract_trend_data(trend_data)
            val, status = get_trend_val(ds, st, vt)
            row[f"NIBP_{component.upper()}"] = val
            row[f"NIBP_{component.upper()}_status"] = status
    else:
        # If NIBP data is completely missing, add entries with unmonitored status
        for component in ["SYS", "DIA", "MAP"]:
            row[f"NIBP_{component}"] = None
            row[f"NIBP_{component}_status"] = "unmonitored"

    # Process IBP readings
    ibp_list = trend.get("Ibp", [])
    for ibI in ibp_list:
        cN = ibI.get("@ChanNum")
        for component in ["Sys", "Dia", "Map"]:
            trend_data = ibI.get(component.lower(), {}).get("TrendData", {})
            ds, st, vt = extract_trend_data(trend_data)
            val, status = get_trend_val(ds, st, vt)
            row[f"IBP{cN}_{component.upper()}"] = val
            row[f"IBP{cN}_{component.upper()}_status"] = status

    return [row]

def parse_trend_rpt_optimized(tr, devS):
    """Optimized version of trend report parsing with status tracking"""
    dt_s = tr.get("StdHdr", {}).get("DevDateTime")
    try:
        dt_o = parser.parse(dt_s)
    except:
        return []

    row = {
        "TimeObj": dt_o,
        "TimeStr": dt_s,
        "DevSerial": devS
    }

    trend = tr.get("Trend", {})
    vital_data = {}
    
    # Process temperature readings with decimal correction
    tmpA = trend.get("Temp", [])
    if not isinstance(tmpA, list):
        tmpA = [tmpA]  # Convert to list if it's a single object
        
    for i, tA in enumerate(tmpA, start=1):
        if not tA:
            continue
        trend_data = tA.get("TrendData", {})
        ds, st, vt = extract_trend_data(trend_data)
        temp_val, temp_status = get_trend_val(ds, st, vt)
        # Always divide temperature by 10 to correct decimal placement
        if temp_val is not None:
            try:
                temp_val = float(temp_val) / 10.0
            except (ValueError, TypeError):
                temp_val = None
                temp_status = "error"
        vital_data[f"Temp{i}"] = temp_val
        vital_data[f"Temp{i}_status"] = temp_status

    # Process vital signs using vectorized operations
    vital_mappings = {
        "Hr": ("Hr", "TrendData"),
        "FiCO2": ("Fico2", "TrendData"),
        "SpO2": ("Spo2", "TrendData"),
        "EtCO2": ("Etco2", "TrendData"),
        "RespRate": ("Resp", "TrendData")
    }

    for col_name, (trend_key, data_key) in vital_mappings.items():
        trend_obj = trend.get(trend_key, {})
        if not trend_obj:
            continue
        trend_data = trend_obj.get(data_key, {})
        ds, st, vt = extract_trend_data(trend_data)
        val, status = get_trend_val(ds, st, vt)
        vital_data[col_name] = val
        vital_data[f"{col_name}_status"] = status

    # Process SpO2-related parameters
    sO2 = trend.get("Spo2", {})
    spo2_params = ["SpMet", "SpCo", "PVI", "PI", "SpOC", "SpHb"]
    for param in spo2_params:
        param_obj = sO2.get(param, {})
        if not param_obj:
            continue
        trend_data = param_obj.get("TrendData", {})
        ds, st, vt = extract_trend_data(trend_data)
        val, status = get_trend_val(ds, st, vt)
        vital_data[param] = val
        vital_data[f"{param}_status"] = status

    # Process NIBP readings with improved handling
    nA = trend.get("Nibp", {})
    if nA:
        for component in ["Sys", "Dia", "Map"]:
            # Try both lowercase and original case to handle inconsistencies in JSON structure
            component_obj = nA.get(component.lower(), nA.get(component, {}))
            if not component_obj:
                continue
                
            trend_data = component_obj.get("TrendData", {})
            ds, st, vt = extract_trend_data(trend_data)
            val, status = get_trend_val(ds, st, vt)
            
            # Map to standardized column names
            bp_key = f"NIBP_{component.upper()}"
            vital_data[bp_key] = val
            vital_data[f"{bp_key}_status"] = status
    else:
        # If NIBP data is completely missing, add entries with unmonitored status
        for component in ["SYS", "DIA", "MAP"]:
            vital_data[f"NIBP_{component}"] = None
            vital_data[f"NIBP_{component}_status"] = "unmonitored"

    # Process IBP readings with improved handling
    ibp_list = trend.get("Ibp", [])
    if not isinstance(ibp_list, list):
        ibp_list = [ibp_list]  # Convert to list if it's a single object
        
    for ibI in ibp_list:
        if not ibI:
            continue
            
        cN = ibI.get("@ChanNum", "1")  # Default to channel 1 if not specified
        
        for component in ["Sys", "Dia", "Map"]:
            component_obj = ibI.get(component.lower(), {})
            if not component_obj:
                continue
                
            trend_data = component_obj.get("TrendData", {})
            ds, st, vt = extract_trend_data(trend_data)
            val, status = get_trend_val(ds, st, vt)
            
            if val is not None:
                vital_data[f"IBP{cN}_{component.upper()}"] = val
                vital_data[f"IBP{cN}_{component.upper()}_status"] = status

    # Update row with all vital data at once
    row.update(vital_data)
    return [row]

def parse_one_json_optimized(json_path):
    """Optimized version of JSON parsing"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        recs = data["ZOLL"]["FullDisclosure"][0]["FullDisclosureRecord"]
    except:
        return []

    # Extract device serial number efficiently
    devS = next((
        rr["DeviceConfiguration"].get("DeviceSerialNumber")
        for rr in recs
        if "DeviceConfiguration" in rr
    ), None)

    # Process trend reports in batches
    all_rows = []
    batch = []
    
    for rr in recs:
        if "TrendRpt" in rr:
            batch.extend(parse_trend_rpt_optimized(rr["TrendRpt"], devS))
            if len(batch) >= CHUNK_SIZE:
                all_rows.extend(batch)
                batch = []
    
    if batch:
        all_rows.extend(batch)
    
    # Add source file information
    base = os.path.basename(json_path)
    for row in all_rows:
        row["SourceFile"] = base
    
    return all_rows

###############################################################################
# PAGE CONFIG
###############################################################################
st.set_page_config(
    page_title="Simulated Patient Physiologic Parameter Analysis",
    page_icon="🤖",
    layout="wide"
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Modern color palette */
:root {
    --primary: #4361ee;
    --primary-light: #4895ef;
    --primary-dark: #3f37c9;
    --accent: #4cc9f0;
    --success: #4CAF50;
    --warning: #ff9e00;
    --danger: #ef476f;
    --neutral-dark: #2b2d42;
    --neutral: #8d99ae;
    --neutral-light: #edf2f4;
    --bg-gradient-start: #f8f9fa;
    --bg-gradient-end: #e9ecef;
}

/* Body & Base Container */
body, .block-container {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background: linear-gradient(135deg, var(--bg-gradient-start) 0%, var(--bg-gradient-end) 100%) !important;
    color: var(--neutral-dark);
    line-height: 1.6;
}

/* Main container with proper spacing */
.block-container {
    padding: 2rem 3rem;
    max-width: 90%;
    margin: 0 auto;
}

/* Make sure no overlay covers our tabs */
body:before {
    content: none !important;
}

/* TABS: Modern, clean styling */
div[data-testid="stTabs"] {
    margin-top: 1rem;
}

div[data-testid="stTabs"] button[data-baseweb="tab"] {
    background-color: transparent !important;
    border: none !important;
    color: var(--neutral) !important;
    border-radius: 0 !important;
    margin-right: 1.5rem !important;
    margin-top: 0.5rem !important;
    padding: 0.8rem 0 !important;
    font-weight: 500;
    position: relative;
    transition: all 0.2s ease;
}

div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
    color: var(--primary) !important;
}

/* Active tab with bottom indicator line */
div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--primary) !important;
    font-weight: 600 !important;
}

div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"]::after {
    content: "";
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background-color: var(--primary);
    border-radius: 3px 3px 0 0;
}

/* Tab content container */
div[data-testid="stTabs"] div[role="tabpanel"] {
    padding: 1rem 0;
}

/* HEADINGS */
h1 {
    color: var(--neutral-dark);
    font-weight: 700;
    font-size: 2.2rem;
    margin-bottom: 1.2rem;
    letter-spacing: -0.02em;
    line-height: 1.2;
}

h2, .stMarkdown h2 {
    color: var(--neutral-dark);
    font-weight: 600;
    font-size: 1.7rem;
    margin: 1.5rem 0 1rem 0;
    letter-spacing: -0.01em;
}

h3, .stMarkdown h3 {
    color: var(--neutral-dark);
    font-weight: 600;
    font-size: 1.3rem;
    margin: 1.2rem 0 0.8rem 0;
}

/* Sections with card styling */
.section-panel {
    background-color: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(8px);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.8rem;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
    border: 1px solid rgba(0, 0, 0, 0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.section-panel:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.06);
}

/* Dataframe styling */
[data-testid="stTable"], .dataframe, div.element-container div.stDataFrame {
    background: rgba(255, 255, 255, 0.6);
    border: 1px solid rgba(0, 0, 0, 0.08);
    border-radius: 8px;
    overflow: hidden;
}

[data-testid="stTable"] table, .dataframe table, div.element-container div.stDataFrame table {
    border-collapse: separate;
    border-spacing: 0;
    width: 100%;
}

[data-testid="stTable"] th, .dataframe th, div.element-container div.stDataFrame th {
    background-color: var(--neutral-light);
    color: var(--neutral-dark);
    font-weight: 600;
    text-align: left;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid rgba(0, 0, 0, 0.1);
}

[data-testid="stTable"] td, .dataframe td, div.element-container div.stDataFrame td {
    padding: 0.6rem 1rem;
    border-bottom: 1px solid rgba(0, 0, 0, 0.05);
}

/* Buttons: modern style */
.stButton > button, button[kind="primary"] {
    background-color: var(--primary) !important;
    color: white !important;
    border: none !important;
    padding: 0.5rem 1.2rem !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 5px rgba(67, 97, 238, 0.2) !important;
}

.stButton > button:hover, button[kind="primary"]:hover {
    background-color: var(--primary-dark) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 8px rgba(67, 97, 238, 0.3) !important;
}

.stButton > button:active, button[kind="primary"]:active {
    transform: translateY(1px) !important;
    box-shadow: 0 1px 3px rgba(67, 97, 238, 0.3) !important;
}

/* Select boxes and inputs */
.stSelectbox div[data-baseweb="select"] > div, 
.stMultiSelect div[data-baseweb="select"] > div,
.stTextInput input {
    background-color: white;
    border: 1px solid rgba(0, 0, 0, 0.1);
    border-radius: 6px !important;
    transition: all 0.2s ease;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
}

.stSelectbox div[data-baseweb="select"] > div:focus-within, 
.stMultiSelect div[data-baseweb="select"] > div:focus-within,
.stTextInput input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px rgba(67, 97, 238, 0.15) !important;
}

/* Radio buttons */
.stRadio > div {
    display: flex;
    gap: 1rem;
}

.stRadio label {
    padding: 0.4rem 1rem;
    background-color: white;
    border: 1px solid rgba(0, 0, 0, 0.1);
    border-radius: 6px;
    transition: all 0.2s ease;
}

.stRadio label:has(input:checked) {
    background-color: var(--primary-light);
    color: white;
    border-color: var(--primary-light);
}

/* Alerts (info, warning, etc.) */
.element-container .stAlert {
    border-radius: 8px;
    border: none !important;
    padding: 1rem 1.2rem;
    display: flex;
    align-items: center;
    background: white;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

.element-container .stAlert.warning {
    border-left: 4px solid var(--warning) !important;
    background: rgba(255, 158, 0, 0.05);
}

.element-container .stAlert.error {
    border-left: 4px solid var(--danger) !important;
    background: rgba(239, 71, 111, 0.05);
}

.element-container .stAlert.info {
    border-left: 4px solid var(--accent) !important;
    background: rgba(76, 201, 240, 0.05);
}

.element-container .stAlert.success {
    border-left: 4px solid var(--success) !important;
    background: rgba(76, 175, 80, 0.05);
}

/* Charts styling */
[data-testid="stArrowVegaLiteChart"], [data-testid="stVegaLiteChart"] {
    background: white;
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    border: 1px solid rgba(0, 0, 0, 0.05);
}

/* Dividers */
hr {
    margin: 2rem 0;
    border: none;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,0,0,0.05) 0%, rgba(0,0,0,0.1) 50%, rgba(0,0,0,0.05) 100%);
}

/* Blockquotes */
blockquote {
    border-left: 4px solid var(--primary-light);
    padding: 0.8rem 1.2rem;
    background-color: rgba(72, 149, 239, 0.05);
    margin: 1.5rem 0;
    border-radius: 0 8px 8px 0;
}

/* Cards for metrics */
.metric-card {
    background: white;
    border-radius: 10px;
    padding: 1.2rem;
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.06);
    transition: transform 0.2s ease;
    border: 1px solid rgba(0, 0, 0, 0.05);
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary);
}

.metric-label {
    color: var(--neutral);
    font-size: 0.9rem;
    font-weight: 500;
}

/* JSON display */
pre {
    background-color: var(--neutral-light);
    border-radius: 8px;
    padding: 1rem;
    overflow-x: auto;
    border: 1px solid rgba(0, 0, 0, 0.05);
}

/* Scrollbars */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.03);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: rgba(0, 0, 0, 0.15);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 0, 0, 0.2);
}

/* Responsive adjustments */
@media screen and (max-width: 768px) {
    .block-container {
        padding: 1rem;
        max-width: 100%;
    }
    
    h1 {
        font-size: 1.8rem;
    }
    
    h2 {
        font-size: 1.5rem;
    }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

###############################################################################
# SCENARIO / THRESHOLDS / SCHEDULE
###############################################################################
SIM_MANNEQUIN_SCENARIO = {
    "Sim1": {"Dave": "TBI", "Chuck": "Sepsis"},
    "Sim2": {"Freddy": "TBI_unmonitored", "Oscar": "DCR"},
    "Sim3": {"Dave": "SepsisARDS", "Chuck": "ACS"},
    "Sim4": {"Dave": "TBI", "Chuck": "AorticDissectionStroke"},
    "Sim5": {"Freddy": "Burn", "Oscar": "AFib", "Matt": "Trauma"},
}

# Scenario-specific monitoring requirements
SCENARIO_REQUIREMENTS = {
    "TBI": {
        "critical_vitals": ["NIBP_MAP", "ICP", "CPP", "Hr", "SpO2", "EtCO2", "TempF", "RespRate"],
        "labs": ["Glucose", "Na", "K"],
        "interventions": ["EVD_zeroing", "sedation_adjustment", "3%_saline", "insulin", "vasopressors"],
        "events": ["ICP_spike", "CPP_low", "sedation_light", "hyperglycemia", "hypernatremia", "fever"]
    },
    "TBI_unmonitored": {
        "critical_vitals": ["NIBP_MAP", "Hr", "SpO2", "EtCO2", "TempF", "RespRate"],
        "labs": ["INR", "PT"],
        "interventions": ["blood_products", "vasopressors", "sedation_adjustment"],
        "events": ["coagulopathy", "hypotension", "under_sedation"]
    },
    "Sepsis": {
        "critical_vitals": ["NIBP_MAP", "Hr", "SpO2", "RespRate", "TempF"],
        "labs": ["Lactate", "WBC", "Creatinine"],
        "interventions": ["antibiotics", "fluid_bolus", "vasopressors", "intubation"],
        "events": ["hypotension", "hypoxia", "fever", "tachycardia"]
    },
    "SepsisARDS": {
        "critical_vitals": ["NIBP_MAP", "Hr", "SpO2", "RespRate", "TempF", "PEEP", "FiO2"],
        "labs": ["K", "Lactate", "ABG"],
        "interventions": ["vent_adjustment", "proning", "antibiotics", "vasopressors"],
        "events": ["hyperkalemia", "hypoxia", "fever", "shock"]
    },
    "ACS": {
        "critical_vitals": ["NIBP_SYS", "NIBP_MAP", "Hr", "SpO2", "RespRate"],
        "labs": ["Troponin", "CK_MB"],
        "interventions": ["defibrillation", "ACLS", "sedation", "cooling"],
        "events": ["v_tach", "ROSC", "bradycardia", "tachycardia"]
    },
    "DCR": {
        "critical_vitals": ["NIBP_SYS", "NIBP_MAP", "Hr", "SpO2", "TempF"],
        "labs": ["Hgb", "INR", "Lactate"],
        "interventions": ["blood_products", "TXA", "calcium", "warming"],
        "events": ["hemorrhage", "coagulopathy", "hypothermia"]
    }
}

CPG_THRESHOLDS = {
    "TBI": {
        "NIBP_SYS": (110, 160), "NIBP_DIA": (60, None), "NIBP_MAP": (80, 100),
        "ICP": (None, 20), "CPP": (60, None),
        "Hr": (60, 100), "RespRate": (12, 20),
        "SpO2": (95, None), "EtCO2": (35, 45), 
        "TempF": (96.8, 99.5),
        "Glucose": (80, 180),
        "Na": (135, 155),
        "sedation_score": (-2, 0)  # RASS target
    },
    "TBI_unmonitored": {
        "NIBP_MAP": (80, 100),
        "Hr": (60, 100),
        "SpO2": (95, None),
        "EtCO2": (35, 45),
        "TempF": (96.8, 99.5)
    },
    "Sepsis": {
        "NIBP_SYS": (90, None), "NIBP_MAP": (65, None),
        "Hr": (None, 100), "RespRate": (12, 24),
        "SpO2": (92, None), "TempF": (96.8, 100.4),
        "Lactate": (None, 2.0),
        "time_to_antibiotics": (None, 60),  # minutes
        "fluid_response": (None, 30)  # minutes to MAP improvement
    },
    "SepsisARDS": {
        "NIBP_SYS": (90, None), "NIBP_MAP": (65, None),
        "Hr": (60, 100), "RespRate": (12, 24),
        "SpO2": (88, 95), "PEEP": (10, None),
        "FiO2": (None, 60), "TempF": (96.8, 100.4),
        "K": (3.5, 5.0), "pH": (7.30, 7.45),
        "PaO2": (55, None), "PaCO2": (35, 50)
    },
    "ACS": {
        "NIBP_SYS": (90, 160), "NIBP_MAP": (65, 110),
        "Hr": (60, 100), "RespRate": (12, 20),
        "SpO2": (94, None), "TempF": (96.8, 99.5),
        "time_to_defib": (None, 2),  # minutes
        "time_to_ROSC": (None, 20)  # minutes
    },
    "AorticDissectionStroke": {
        "NIBP_SYS": (100, 120), "NIBP_DIA": (60, 80),
        "Hr": (60, 80), "RespRate": (12, 20),
        "SpO2": (92, None), "TempF": (96.8, 99.5),
        "time_to_bp_control": (None, 30)  # minutes
    },
    "Burn": {
        "NIBP_SYS": (90, 160), "NIBP_MAP": (65, None),
        "Hr": (60, 120), "RespRate": (12, 24),
        "SpO2": (92, None), "TempF": (96.8, 103),
        "parkland_fluid_rate": (None, None),  # Special handling
        "urine_output": (0.5, None)  # mL/kg/hr
    },
    "AFib": {
        "NIBP_SYS": (90, 140), "NIBP_MAP": (65, None),
        "Hr": (60, 100), "RespRate": (12, 20),
        "SpO2": (92, None), "TempF": (96.8, 99.5),
        "time_to_rate_control": (None, 60)  # minutes
    },
    "Trauma": {
        "NIBP_SYS": (90, None), "NIBP_MAP": (65, None),
        "Hr": (60, 120), "RespRate": (12, 24),
        "SpO2": (92, None), "TempF": (96.8, 99.5),
        "Hgb": (7, None), "INR": (None, 1.5),
        "time_to_hemorrhage_control": (None, 30)  # minutes
    },
    "DCR": {
        "NIBP_SYS": (90, None),
        "NIBP_MAP": (65, None),
        "Hr": (60, 120),
        "SpO2": (92, None),
        "TempF": (96.8, 99.5),
        "Hgb": (7, None),
        "INR": (None, 1.5),
        "Lactate": (None, 4.0)
    }
}

SIM_MANNEQUINS = {
    "Sim1": ["Dave","Chuck"],
    "Sim2": ["Freddy","Oscar"],
    "Sim3": ["Dave","Chuck"],
    "Sim4": ["Dave","Chuck"],
    "Sim5": ["Freddy","Oscar","Matt"]
}

MANNEQUIN_MAP = {
    "AI23F013939": "Dave",
    "AI23H014090": "Chuck",
    "AI15F004305": "Freddy",
    "AI15D003889": "Matt",
    "AI20C009617": "Oscar"
}

def convert_time(hhmm):
    s = str(hhmm)
    if len(s)<=2:
        return time(int(s),0)
    hh = int(s[:-2])
    mm = int(s[-2:])
    return time(hh, mm)

SIM_SCHEDULES = {
    "Sim1": [(convert_time(830), convert_time(955)), (convert_time(955), convert_time(1120)), (convert_time(1120), convert_time(1245)), (convert_time(1305), convert_time(1430)), (convert_time(1430), convert_time(1555)), (convert_time(1555), convert_time(1720))],
    "Sim2": [(convert_time(800), convert_time(920)), (convert_time(920), convert_time(1040)), (convert_time(1040), convert_time(1200)), (convert_time(1220), convert_time(1340)), (convert_time(1340), convert_time(1500)), (convert_time(1500), convert_time(1620))],
    "Sim3": [(convert_time(800), convert_time(930)), (convert_time(930), convert_time(1100)), (convert_time(1100), convert_time(1230)), (convert_time(1250), convert_time(1420)), (convert_time(1420), convert_time(1550)), (convert_time(1550), convert_time(1720))],
    "Sim4": [(convert_time(800), convert_time(925)), (convert_time(925), convert_time(1050)), (convert_time(1050), convert_time(1215)), (convert_time(1235), convert_time(1400)), (convert_time(1400), convert_time(1525)), (convert_time(1525), convert_time(1650))],
    "Sim5": [(convert_time(800), convert_time(910)), (convert_time(910), convert_time(1020)), (convert_time(1020), convert_time(1130)), (convert_time(1150), convert_time(1300)), (convert_time(1300), convert_time(1410)), (convert_time(1410), convert_time(1520))]
}

# Add after SIM_SCHEDULES definition
SIM_DURATIONS = {
    "Sim1": 45,  # 45 minute simulation
    "Sim2": 35,  # 35 minute simulation (7min prep + 10min ground + 15min flight)
    "Sim3": 50,  # 50 minute simulation
    "Sim4": 55,  # 55 minute simulation (includes 5min handoff)
    "Sim5": 40,  # 40 minute simulation (10min initial + 10min ground + 13min flight)
}

COURSE_DATE_RANGES = [
    ("10/14","10/18","2025A"),
    ("10/28","11/01","2025B"),
    ("11/18","11/22","2025C"),
    ("12/09","12/13","2025D"),
    ("01/13","01/17","2025E"),
    ("01/27","01/31","2025F"),
    ("02/17","02/21","2025G/H"),
    ("02/24","02/28","2025H"),
    ("03/24","03/28","2025I"),
    ("04/14","04/18","2025J"),
    ("05/05","05/09","2025K/L"),
    ("05/12","05/16","2025L"),
    ("06/09","06/13","2025M"),
    ("06/23","06/27","2025N"),
]

COURSE_START_DATES = {
    "2025A": "2024-10-14","2025B": "2024-10-28","2025C": "2024-11-18","2025D": "2024-12-09",
    "2025E": "2025-01-13","2025F": "2025-01-27","2025G/H": "2025-02-17","2025H": "2025-02-24",
    "2025I": "2025-03-24","2025J": "2025-04-14","2025K/L": "2025-05-05","2025L": "2025-05-12",
    "2025M": "2025-06-09","2025N": "2025-06-23"
}

DAY_OFFSET_TO_SIM = {0:"Sim1",1:"Sim2",2:"Sim3",3:"Sim4",4:"Sim5"}

def parse_mmdd(s):
    m,d = s.split("/")
    return int(m), int(d)

def get_course_for_date(dt_obj):
    fm, fd = dt_obj.month, dt_obj.day
    for (start_s, end_s, cname) in COURSE_DATE_RANGES:
        sm, sd = parse_mmdd(start_s)
        em, ed = parse_mmdd(end_s)
        ok_start = (fm>sm) or (fm==sm and fd>=sd)
        ok_end   = (fm<em) or (fm==em and fd<=ed)
        if ok_start and ok_end:
            return cname
    return None

def get_sim_for_course_date(course, dt_obj):
    if not course or course not in COURSE_START_DATES:
        return None
    try:
        cstart = datetime.strptime(COURSE_START_DATES[course], "%Y-%m-%d")
    except:
        return None
    diff = (dt_obj.date() - cstart.date()).days
    return DAY_OFFSET_TO_SIM.get(diff)

def in_sim_time_window(dt_obj, sim):
    """
    Enhanced version that considers actual simulation duration and filters setup/cleanup periods.
    Each simulation block has:
    - Setup period (10 minutes)
    - Actual simulation time (varies by simulation)
    - Cleanup/debrief period (remainder of block)
    """
    if sim not in SIM_SCHEDULES:
        return False
        
    the_time = dt_obj.time()
    sim_duration = SIM_DURATIONS.get(sim, 0)  # Get planned duration in minutes
    
    for (startT, endT) in SIM_SCHEDULES[sim]:
        # Calculate setup buffer and actual simulation window
        setup_buffer = 10  # 10 minutes for setup
        
        # Calculate the actual simulation start time (after setup buffer)
        sim_start_minutes = startT.hour * 60 + startT.minute + setup_buffer
        sim_start = time(sim_start_minutes // 60, sim_start_minutes % 60)
        
        # Calculate the simulation end time (sim_start + duration)
        sim_end_minutes = sim_start_minutes + sim_duration
        sim_end = time(sim_end_minutes // 60, sim_end_minutes % 60)
        
        # Check if current time falls within the actual simulation period
        if sim_start <= the_time <= sim_end:
            return True
            
    return False

###############################################################################
# PARSING JSON => ROWS
###############################################################################
def parse_one_json(json_path):
    """Parse a single JSON file"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        recs = data["ZOLL"]["FullDisclosure"][0]["FullDisclosureRecord"]
    except:
        return []

    # Extract device serial number efficiently
    devS = next((
        rr["DeviceConfiguration"].get("DeviceSerialNumber")
        for rr in recs
        if "DeviceConfiguration" in rr
    ), None)

    # Process trend reports in batches
    all_rows = []
    batch = []
    
    for rr in recs:
        if "TrendRpt" in rr:
            batch.extend(parse_trend_rpt(rr["TrendRpt"], devS))
            if len(batch) >= CHUNK_SIZE:
                all_rows.extend(batch)
                batch = []
    
    if batch:
        all_rows.extend(batch)
    
    # Add source file information
    base = os.path.basename(json_path)
    for row in all_rows:
        row["SourceFile"] = base
    
    return all_rows

def is_valid_vectorized(df):
    """Vectorized version of validity checking"""
    mask = pd.notnull(df["Course"]) & pd.notnull(df["Sim"])
    mask &= (df["InSimWindow"] == True)
    mask &= pd.notnull(df["overrideMannequin"])
    
    # Create a mapping Series for valid mannequins per sim
    valid_mannequins = {sim: set(mannequins) for sim, mannequins in SIM_MANNEQUINS.items()}
    mask &= df.apply(lambda row: row["overrideMannequin"] in valid_mannequins.get(row["Sim"], []), axis=1)
    
    return mask

def pick_scenario_vectorized(df):
    """Vectorized version of scenario selection"""
    # Create a mapping dictionary for scenarios
    scenario_map = {}
    for sim, scenarios in SIM_MANNEQUIN_SCENARIO.items():
        for mannequin, scenario in scenarios.items():
            scenario_map[(sim, mannequin)] = scenario
    
    # Create a Series of tuples (Sim, overrideMannequin)
    sim_man_pairs = list(zip(df["Sim"], df["overrideMannequin"]))
    
    # Map scenarios using the dictionary, defaulting to "General"
    return pd.Series([scenario_map.get(pair, "General") for pair in sim_man_pairs], index=df.index)

def attempt_override_vectorized(subdf):
    """Vectorized version of mannequin override"""
    if subdf.empty:
        return subdf
        
    sim = subdf["Sim"].iloc[0]
    correct_set = set(SIM_MANNEQUINS.get(sim, []))
    if not correct_set:
        return subdf
        
    # Work on the rows within sim window
    mask = subdf["InSimWindow"] == True
    if not mask.any():
        return subdf
        
    fixable = subdf[mask].copy()
    used = set(fixable["rawMannequin"].dropna().unique())
    missing = list(correct_set - used)
    
    if not missing:
        return subdf
        
    # Find invalid mannequins
    invalid_mask = ~fixable["rawMannequin"].isin(correct_set) & fixable["rawMannequin"].notna()
    invalid_indices = fixable[invalid_mask].index
    
    # Assign missing mannequins to invalid entries
    for idx, new_mannequin in zip(invalid_indices, missing):
        fixable.at[idx, "overrideMannequin"] = new_mannequin
    
    # Combine and sort
    result = pd.concat([fixable, subdf[~mask]], ignore_index=True)
    return result.sort_values("TimeObj")

def load_and_sort_data(folder_path):
    """Optimized data loading with parallel processing and caching"""
    # Check cache first
    cache_key = get_cache_key(folder_path)
    cached_df = load_from_cache(cache_key)
    if cached_df is not None:
        return cached_df

    import glob
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    # Process files in parallel with optimized worker count
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(parse_one_json_optimized, json_files))
    
    # Efficient list comprehension for flattening
    big_rows = [row for result in results for row in result]
    if not big_rows:
        return pd.DataFrame()

    # Create DataFrame with optimized dtypes
    df = pd.DataFrame(big_rows)
    
    # Optimize data types using categorical and efficient numeric conversions
    categorical_columns = ["DevSerial", "SourceFile", "rawMannequin", "Course", 
                         "Sim", "overrideMannequin", "DateStr", "Scenario"]
    
    # Convert categoricals in batch
    for col in categorical_columns:
        if col in df.columns:
            df[col] = pd.Categorical(df[col])

    # Vectorized numeric conversion
    numeric_cols = [col for col in df.columns if col not in categorical_columns + ["TimeObj", "TimeStr"]]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce", downcast='float')

    if "TimeObj" in df.columns:
        # Efficient sorting and processing
        df = df.sort_values("TimeObj").reset_index(drop=True)

        # Vectorized operations for derived columns
        df["rawMannequin"] = pd.Categorical(df["DevSerial"].map(MANNEQUIN_MAP))
        df["Course"] = pd.Categorical(df["TimeObj"].apply(get_course_for_date))
        
        # Optimize Sim and window calculations
        df["Sim"] = pd.Categorical(df.apply(lambda row: get_sim_for_course_date(row["Course"], row["TimeObj"]), axis=1))
        
        # Apply enhanced simulation window filtering
        df["InSimWindow"] = df.apply(lambda row: in_sim_time_window(row["TimeObj"], row["Sim"]) 
                                   if row["Sim"] else False, axis=1)
        
        df["overrideMannequin"] = df["rawMannequin"]
        df["DateStr"] = df["TimeObj"].dt.strftime("%Y-%m-%d")

        # Process mannequin overrides efficiently
        groups = [attempt_override_vectorized(group) 
                 for _, group in df.groupby(["Course", "Sim", "DateStr"])]
        
        df = pd.concat(groups, ignore_index=True)
        df = df.sort_values("TimeObj")

        # Vectorized validity and scenario checks
        df["IsValid"] = is_valid_vectorized(df)
        df["Scenario"] = pd.Categorical(pick_scenario_vectorized(df))

        # Cache with optimized settings
        save_to_cache(df, cache_key)

    return df

def compute_time_out_of_range(subdf, scenario, original_scenario=None):
    """Enhanced version of compliance calculation with scenario-specific requirements"""
    if subdf.empty:
        return {
            "scenario": scenario,
            "original_scenario": original_scenario,
            "total_duration_min": 0,
            "compliance_score": 0,
            "vital_stats": {},
            "events": ["WARNING: No data available for analysis"],
            "summary": "No data available for analysis",
            "assessment": "POOR - No data available for analysis"
        }

    # Add debug information
    events = []
    events.append(f"DEBUG: Processing scenario '{scenario}' with {len(subdf)} data points")
    
    # Check if scenario exists in thresholds
    if scenario not in CPG_THRESHOLDS:
        events.append(f"CRITICAL: Scenario '{scenario}' not found in CPG_THRESHOLDS")
        return {
            "scenario": scenario,
            "original_scenario": original_scenario,
            "total_duration_min": 0,
            "compliance_score": 0,
            "vital_stats": {},
            "events": events,
            "summary": f"Error: Scenario '{scenario}' not defined",
            "assessment": f"POOR - Scenario '{scenario}' not defined in CPG thresholds"
        }
    
    # Sort and calculate time deltas
    subdf = subdf.sort_values("TimeObj").reset_index(drop=True)
    subdf["delta_sec"] = subdf["TimeObj"].diff().shift(-1).dt.total_seconds().fillna(0)
    
    # Detect multiple simulation runs by looking for gaps > 10 minutes
    large_gaps = subdf[subdf["delta_sec"] > 600]["TimeObj"].tolist()
    if large_gaps:
        events.extend([f"NOTE: Multiple simulation runs detected. Gap at {t.strftime('%H:%M:%S')}" for t in large_gaps])

    total_duration_sec = (subdf["TimeObj"].iloc[-1] - subdf["TimeObj"].iloc[0]).total_seconds()
    total_duration_min = total_duration_sec / 60
    events.append(f"DEBUG: Total duration: {total_duration_min:.1f} minutes")

    thresholds = CPG_THRESHOLDS.get(scenario, {})
    requirements = SCENARIO_REQUIREMENTS.get(scenario, {})
    
    # Debug thresholds
    events.append(f"DEBUG: Found {len(thresholds)} thresholds for scenario '{scenario}'")
    
    # List of special non-vital thresholds that should be skipped
    special_thresholds = {
        'time_to_antibiotics', 'time_to_defib', 'time_to_ROSC', 
        'time_to_bp_control', 'time_to_rate_control', 
        'time_to_hemorrhage_control', 'parkland_fluid_rate',
        'fluid_response', 'sedation_score', 'urine_output'
    }
    
    # Debug available columns
    available_columns = set(subdf.columns)
    vital_columns = [col for col in thresholds.keys() if col not in special_thresholds]
    events.append(f"DEBUG: Available columns: {', '.join(sorted(available_columns))}")
    events.append(f"DEBUG: Looking for vital columns: {', '.join(sorted(vital_columns))}")
    
    vital_stats = {}
    total_compliance_score = 0
    num_vitals_monitored = 0

    # Process each vital sign threshold
    for vital, threshold in thresholds.items():
        # Skip special thresholds
        if vital in special_thresholds:
            continue
            
        # Get min/max thresholds
        vmin, vmax = threshold

        if vital not in subdf.columns:
            if vital in requirements.get('critical_vitals', []):
                events.append(f"WARNING: Critical vital {vital} not monitored")
            continue

        # Debug vital sign data
        events.append(f"DEBUG: Processing vital {vital} with threshold ({vmin}, {vmax})")
        
        # Get the status column name
        status_col = f"{vital}_status"
        
        # Convert to numeric and identify valid measurements
        col_data = pd.to_numeric(subdf[vital], errors="coerce")
        measured_mask = col_data.notna()
        
        # Debug vital sign data
        valid_count = measured_mask.sum()
        events.append(f"DEBUG: Found {valid_count} non-null values for {vital}")
        
        # If we have status information, use it to refine the measured mask
        if status_col in subdf.columns:
            # Debug status information
            status_counts = subdf[status_col].value_counts().to_dict()
            events.append(f"DEBUG: Status counts for {vital}: {status_counts}")
            
            # Only consider values with valid status (more flexible validation)
            # Accept "valid" or any non-error status if no "valid" status exists
            if "valid" in status_counts:
                valid_status_mask = (subdf[status_col] == "valid")
            else:
                # If no "valid" status, accept anything that's not explicitly an error status
                error_statuses = ["error", "invalid", "unmonitored"]
                valid_status_mask = ~subdf[status_col].isin(error_statuses)
                events.append(f"DEBUG: No 'valid' status found for {vital}, using non-error statuses")
            
            measured_mask &= valid_status_mask
            
            # Log periods of different states
            for state in ["unmonitored", "invalid", "error"]:
                state_time = (subdf[status_col] == state).sum() * subdf["delta_sec"].mean() / 60
                if state_time > 1:  # Only log if more than 1 minute
                    events.append(f"NOTE: {vital} was {state} for {round(state_time,1)} minutes")
        else:
            events.append(f"DEBUG: No status column found for {vital}, using all non-null values")

        # Debug final valid measurements
        final_valid_count = measured_mask.sum()
        events.append(f"DEBUG: Final valid count for {vital}: {final_valid_count}")
        
        measured_sec = (measured_mask * subdf["delta_sec"]).sum()

        if measured_sec == 0:
            if vital in requirements.get('critical_vitals', []):
                events.append(f"WARNING: No valid readings for {vital}")
            continue

        num_vitals_monitored += 1
        
        # Calculate out-of-range time and statistics
        range_mask = pd.Series(True, index=subdf.index)
        threshold_desc = []
        
        if vmin is not None:
            range_mask &= (col_data >= vmin)
            threshold_desc.append(f"≥{vmin}")
            # Check for critical low values - only during measured periods
            critical_low_mask = (col_data < vmin) & measured_mask
            if critical_low_mask.any():
                critical_time = subdf[critical_low_mask]["TimeObj"].iloc[0]
                events.append(f"CRITICAL: {vital} below {vmin} at {critical_time.strftime('%H:%M:%S')}")
        
        if vmax is not None:
            range_mask &= (col_data <= vmax)
            threshold_desc.append(f"≤{vmax}")
            # Check for critical high values - only during measured periods
            critical_high_mask = (col_data > vmax) & measured_mask
            if critical_high_mask.any():
                critical_time = subdf[critical_high_mask]["TimeObj"].iloc[0]
                events.append(f"CRITICAL: {vital} above {vmax} at {critical_time.strftime('%H:%M:%S')}")

        # Calculate statistics - only for measured periods
        out_sec = (~range_mask & measured_mask * subdf["delta_sec"]).sum()
        in_range_percent = 100 * (1 - out_sec / measured_sec)
        
        # Calculate monitoring coverage
        coverage_percent = round(100 * measured_sec / total_duration_sec, 1)
        
        # Determine compliance status
        if in_range_percent >= 90:
            status = "Excellent"
        elif in_range_percent >= 75:
            status = "Good"
        elif in_range_percent >= 60:
            status = "Fair"
        else:
            status = "Poor"

        # Calculate detailed statistics for measured periods only
        valid_data = col_data[measured_mask]
        if not valid_data.empty:
            mean_val = round(float(valid_data.mean()), 1)
            min_val = round(float(valid_data.min()), 1)
            max_val = round(float(valid_data.max()), 1)
            range_str = f"{min_val}-{max_val}"
        else:
            mean_val = None
            range_str = "N/A"

        # Store comprehensive vital statistics
        vital_stats[vital] = {
            "in_range_percent": round(in_range_percent, 1),
            "coverage_percent": coverage_percent,
            "status": status,
            "target": " and ".join(threshold_desc) if threshold_desc else "Monitoring only",
            "mean_value": mean_val,
            "range": range_str,
            "time_out_of_range_min": round(out_sec / 60, 1),
            "time_measured_min": round(measured_sec / 60, 1)
        }
        
        total_compliance_score += in_range_percent

    # Calculate overall compliance score
    overall_score = round(total_compliance_score / num_vitals_monitored if num_vitals_monitored > 0 else 0, 1)
    
    # Check if we have any vital stats
    if num_vitals_monitored == 0:
        events.append("CRITICAL: No valid vital signs were monitored or all readings were invalid")
        assessment = "POOR - No valid vital sign data available for compliance assessment"
    else:
        # Generate scenario-specific assessment
        if overall_score >= 90:
            assessment = "EXCELLENT - CPG guidelines were consistently met"
        elif overall_score >= 75:
            assessment = "GOOD - CPG guidelines were generally followed with some deviations"
        elif overall_score >= 60:
            assessment = "FAIR - Significant deviations from CPG guidelines"
        else:
            assessment = "POOR - Major deviations from CPG guidelines"

    # Add scenario override note if applicable
    if original_scenario and original_scenario != scenario:
        events.insert(0, f"NOTE: Scenario overridden from {original_scenario} to {scenario}")

    # Add scenario-specific context and recommendations
    scenario_context = get_scenario_context(scenario)
    
    return {
        "scenario": scenario,
        "original_scenario": original_scenario,
        "total_duration_min": round(total_duration_min, 1),
        "compliance_score": overall_score,
        "assessment": assessment + scenario_context,
        "vital_stats": vital_stats,
        "events": events
    }

def get_scenario_context(scenario):
    """Helper function to get scenario-specific context"""
    contexts = {
        "TBI": (
            "\n\nTBI Management Guidelines:\n"
            "- MAP target 80-100 mmHg for CPP ≥60\n"
            "- ICP target <20 mmHg\n"
            "- Temperature 96.8-99.5°F\n"
            "- Glucose 80-180 mg/dL\n"
            "- Na 135-155 mEq/L"
        ),
        "TBI_unmonitored": (
            "\n\nTBI Management Guidelines (Unmonitored):\n"
            "- MAP target 80-100 mmHg\n"
            "- Temperature 96.8-99.5°F\n"
            "- SpO2 ≥95%\n"
            "- EtCO2 35-45 mmHg\n"
            "- Heart Rate 60-100 bpm"
        ),
        "Sepsis": (
            "\n\nSepsis Management Guidelines:\n"
            "- MAP target ≥65 mmHg\n"
            "- Heart Rate target ≤100 bpm\n"
            "- SpO2 ≥92%\n"
            "- Temperature 96.8-100.4°F\n"
            "- Lactate goal <2.0 mmol/L\n"
            "- Time to antibiotics ≤60 minutes\n"
            "- Fluid response within 30 minutes"
        ),
        "SepsisARDS": (
            "\n\nSepsis with ARDS Management Guidelines:\n"
            "- MAP target ≥65 mmHg\n"
            "- Heart Rate 60-100 bpm\n"
            "- SpO2 88-95%\n"
            "- PEEP ≥10 cmH2O\n"
            "- FiO2 ≤60%\n"
            "- Temperature 96.8-100.4°F\n"
            "- K+ 3.5-5.0 mEq/L\n"
            "- pH 7.30-7.45\n"
            "- PaO2 ≥55 mmHg\n"
            "- PaCO2 35-50 mmHg"
        ),
        "ACS": (
            "\n\nAcute Coronary Syndrome Guidelines:\n"
            "- SBP 90-160 mmHg\n"
            "- MAP 65-110 mmHg\n"
            "- Heart Rate 60-100 bpm\n"
            "- SpO2 ≥94%\n"
            "- Temperature 96.8-99.5°F\n"
            "- Time to defibrillation ≤2 minutes\n"
            "- Time to ROSC ≤20 minutes"
        ),
        "AorticDissectionStroke": (
            "\n\nAortic Dissection/Stroke Guidelines:\n"
            "- SBP 100-120 mmHg\n"
            "- DBP 60-80 mmHg\n"
            "- Heart Rate 60-80 bpm\n"
            "- SpO2 ≥92%\n"
            "- Temperature 96.8-99.5°F\n"
            "- Time to BP control ≤30 minutes"
        ),
        "Burn": (
            "\n\nBurn Management Guidelines:\n"
            "- SBP 90-160 mmHg\n"
            "- MAP ≥65 mmHg\n"
            "- Heart Rate 60-120 bpm\n"
            "- SpO2 ≥92%\n"
            "- Temperature 96.8-103°F\n"
            "- Parkland formula for fluid resuscitation\n"
            "- Urine output ≥0.5 mL/kg/hr"
        ),
        "AFib": (
            "\n\nAtrial Fibrillation Guidelines:\n"
            "- SBP 90-140 mmHg\n"
            "- MAP ≥65 mmHg\n"
            "- Heart Rate 60-100 bpm\n"
            "- SpO2 ≥92%\n"
            "- Temperature 96.8-99.5°F\n"
            "- Time to rate control ≤60 minutes"
        ),
        "Trauma": (
            "\n\nTrauma Management Guidelines:\n"
            "- SBP ≥90 mmHg\n"
            "- MAP ≥65 mmHg\n"
            "- Heart Rate 60-120 bpm\n"
            "- SpO2 ≥92%\n"
            "- Temperature 96.8-99.5°F\n"
            "- Hemoglobin ≥7 g/dL\n"
            "- INR ≤1.5\n"
            "- Time to hemorrhage control ≤30 minutes"
        ),
        "DCR": (
            "\n\nDamage Control Resuscitation Guidelines:\n"
            "- SBP ≥90 mmHg\n"
            "- MAP ≥65 mmHg\n"
            "- Heart Rate 60-120 bpm\n"
            "- SpO2 ≥92%\n"
            "- Temperature 96.8-99.5°F\n"
            "- Hemoglobin ≥7 g/dL\n"
            "- INR ≤1.5\n"
            "- Lactate monitoring\n"
            "- Balanced blood product resuscitation\n"
            "- Early hemorrhage control"
        )
    }
    return contexts.get(scenario, "\n\nNo specific guidelines available for this scenario.")

###############################################################################
# MAIN APP
###############################################################################
def check_and_fix_vital_data(df, scenario):
    """
    Check if vital sign data is available for the given scenario and fix if possible.
    Returns a tuple of (fixed_df, messages)
    """
    messages = []
    
    # Get the required vital signs for this scenario
    thresholds = CPG_THRESHOLDS.get(scenario, {})
    if not thresholds:
        messages.append(f"ERROR: No thresholds defined for scenario '{scenario}'")
        return df, messages
    
    # Check which vital signs are available
    vital_columns = [col for col in thresholds.keys() if col in df.columns]
    if not vital_columns:
        messages.append(f"ERROR: None of the required vital signs for '{scenario}' are present in the data")
        return df, messages
    
    messages.append(f"Found {len(vital_columns)} vital signs for scenario '{scenario}': {', '.join(vital_columns)}")
    
    # Check for common naming issues and fix them
    vital_mapping = {
        "TempF": ["Temp1", "Temp", "Temperature"],  # Try to map TempF to Temp1 or Temp
        "Hr": ["HR", "HeartRate", "Heart_Rate", "Pulse"],    # Try to map Hr to HR or HeartRate
        "SpO2": ["SPO2", "Spo2", "O2Sat", "Oxygen", "SaO2"],     # Try to map SpO2 to SPO2 or Spo2
        "NIBP_SYS": ["NIBP_Sys", "SBP", "Systolic", "NIBP_SYSTOLIC", "BP_Systolic", "Sys"],  # Map NIBP_SYS to alternatives
        "NIBP_DIA": ["NIBP_Dia", "DBP", "Diastolic", "NIBP_DIASTOLIC", "BP_Diastolic", "Dia"],  # Map NIBP_DIA to alternatives
        "NIBP_MAP": ["NIBP_Map", "MAP", "MeanArterialPressure", "NIBP_MEAN", "Mean", "BP_Mean"]  # Map NIBP_MAP to alternatives
    }
    
    fixed_df = df.copy()
    
    # Try to fix missing vital signs
    for target_vital, alternative_names in vital_mapping.items():
        if target_vital in thresholds and target_vital not in df.columns:
            # Try each alternative name
            for alt_name in alternative_names:
                if alt_name in df.columns:
                    fixed_df[target_vital] = df[alt_name]
                    fixed_df[f"{target_vital}_status"] = df.get(f"{alt_name}_status", "valid")
                    messages.append(f"Mapped missing vital '{target_vital}' to existing column '{alt_name}'")
                    break
    
    # Special handling for blood pressure
    # If we have IBP but not NIBP, use IBP as a fallback
    for bp_component in ["SYS", "DIA", "MAP"]:
        nibp_col = f"NIBP_{bp_component}"
        
        # Check if we need this vital and it's missing or has no valid readings
        if nibp_col in thresholds and (
            nibp_col not in fixed_df.columns or 
            f"{nibp_col}_status" not in fixed_df.columns or
            fixed_df[fixed_df[f"{nibp_col}_status"] == "valid"].empty
        ):
            # First try to find alternative NIBP column names with different case
            for alt_nibp in [f"NIBP_{bp_component.lower()}", f"NIBP_{bp_component.capitalize()}", 
                            f"Nibp_{bp_component}", f"nibp_{bp_component}"]:
                if alt_nibp in fixed_df.columns:
                    fixed_df[nibp_col] = fixed_df[alt_nibp]
                    status_col = f"{alt_nibp}_status" if f"{alt_nibp}_status" in fixed_df.columns else None
                    if status_col:
                        fixed_df[f"{nibp_col}_status"] = fixed_df[status_col]
                    else:
                        # If no status column, assume values are valid if not null
                        fixed_df[f"{nibp_col}_status"] = "valid"
                        fixed_df.loc[fixed_df[nibp_col].isna(), f"{nibp_col}_status"] = "unmonitored"
                    messages.append(f"Mapped {alt_nibp} to standard column name {nibp_col}")
                    break
            
            # If still missing, try to use IBP as fallback
            if nibp_col not in fixed_df.columns or fixed_df[fixed_df[f"{nibp_col}_status"] == "valid"].empty:
                for ibp_prefix in ["IBP1_", "IBP2_", "IBP3_", "IBP_", "ART_"]:
                    ibp_col = f"{ibp_prefix}{bp_component}"
                    if ibp_col in fixed_df.columns:
                        # Check if there are valid readings
                        if f"{ibp_col}_status" in fixed_df.columns:
                            valid_readings = fixed_df[fixed_df[f"{ibp_col}_status"] == "valid"]
                            if not valid_readings.empty:
                                fixed_df[nibp_col] = fixed_df[ibp_col]
                                fixed_df[f"{nibp_col}_status"] = fixed_df[f"{ibp_col}_status"]
                                messages.append(f"Using {ibp_col} as fallback for {nibp_col}")
                                break
                        else:
                            # If no status column, assume values are valid if not null
                            valid_readings = fixed_df[fixed_df[ibp_col].notna()]
                            if not valid_readings.empty:
                                fixed_df[nibp_col] = fixed_df[ibp_col]
                                fixed_df[f"{nibp_col}_status"] = "valid"
                                messages.append(f"Using {ibp_col} as fallback for {nibp_col} (assuming valid status)")
                                break
    
    # Special handling for temperature conversion
    if "TempF" in thresholds and "TempF" not in fixed_df.columns:
        # Try to convert from Celsius to Fahrenheit
        for temp_col in ["Temp1", "Temp", "TempC"]:
            if temp_col in fixed_df.columns:
                # Check if values are likely in Celsius (below 50)
                temp_values = pd.to_numeric(fixed_df[temp_col], errors="coerce")
                if temp_values.median() < 50:  # Likely Celsius
                    fixed_df["TempF"] = temp_values * 9/5 + 32
                    fixed_df["TempF_status"] = fixed_df.get(f"{temp_col}_status", "valid")
                    messages.append(f"Converted {temp_col} from Celsius to Fahrenheit")
                    break
                else:  # Already in Fahrenheit
                    fixed_df["TempF"] = temp_values
                    fixed_df["TempF_status"] = fixed_df.get(f"{temp_col}_status", "valid")
                    messages.append(f"Copied {temp_col} to TempF (already in Fahrenheit)")
                    break
    
    # Check for vital signs with no valid readings
    for vital in list(thresholds.keys()):
        if vital not in fixed_df.columns:
            continue
            
        status_col = f"{vital}_status"
        
        # If status column doesn't exist, create it with all valid values
        if status_col not in fixed_df.columns:
            fixed_df[status_col] = "valid"
            messages.append(f"Created missing status column for {vital}")
            continue
            
        valid_count = (fixed_df[status_col] == "valid").sum()
        if valid_count == 0:
            # Try to fix by accepting any non-null value
            fixed_df[status_col] = fixed_df[status_col].fillna("valid")
            fixed_df.loc[fixed_df[vital].notna(), status_col] = "valid"
            new_valid_count = (fixed_df[status_col] == "valid").sum()
            messages.append(f"Fixed '{vital}' status: changed {new_valid_count} readings to 'valid'")
    
    # Add debug information about available columns
    all_columns = set(fixed_df.columns)
    vital_related_columns = [col for col in all_columns if any(vs in col for vs in ["NIBP", "IBP", "Temp", "SpO2", "Hr", "EtCO2"])]
    messages.append(f"Available vital-related columns: {', '.join(sorted(vital_related_columns))}")
    
    return fixed_df, messages

def clear_cache():
    """
    Clear all types of cache used by the application:
    1. File-based cache in CACHE_DIR
    2. lru_cache for decorated functions
    """
    # Clear file-based cache
    if os.path.exists(CACHE_DIR):
        try:
            shutil.rmtree(CACHE_DIR)
            os.makedirs(CACHE_DIR, exist_ok=True)
            st.success(f"File cache cleared successfully. Recreated {CACHE_DIR} directory.")
        except Exception as e:
            st.error(f"Error clearing file cache: {str(e)}")
    
    # Clear lru_cache for decorated functions
    try:
        calculate_file_hash.cache_clear()
        get_trend_val.cache_clear()
        st.success("Function cache cleared successfully.")
    except Exception as e:
        st.error(f"Error clearing function cache: {str(e)}")
    
    return True

def main():
    # Initialize database
    init_database()

    tabs = st.tabs(["Home", "Data Explorer", "CPG Compliance", "Visualization", "Real-Time Monitor"])

    ###########################################################################
    # HOME
    ###########################################################################
    with tabs[0]:
        st.title("Simulated Patient Physiologic Parameter Analysis (Demo)")

        st.write("""
        Welcome!
        This prototype application demonstrates how we can analyze physiologic data 
        (e.g., from Zoll Propaq) for **CCAT Advanced validation simulations**. 
        The goal is to provide near real-time feedback on **compliance with Clinical 
        Practice Guidelines (CPGs)**, identify trends in vital sign management, 
        and ultimately improve **training outcomes**.
        """)

        st.write("---")
        st.subheader("Key Features")
        st.write("""
        - **Data Ingestion & Explorer**: Load Propaq JSON files, filter by date/time, and see relevant descriptive stats.
        - **CPG Compliance**: Quickly compute how many seconds your team spent outside target ranges for TBI, Sepsis, Burn, or other scenarios.
        - **Visualization**: Graph trends (line, bar, etc.) to see how vitals changed over time.
        - **Database Storage**: Store and access data without needing local file access.

        > **Note**: This is a demonstration. Future enhancements would tie directly to Wi-Fi streaming, automatically ingest data, and provide real-time feedback mid-simulation!
        """)

        # Add cache clearing button
        if st.button("Clear Application Cache"):
            clear_cache()
            st.info("Please reload the application for changes to take effect.")

    ###########################################################################
    # TAB 1: DATA EXPLORER
    ###########################################################################
    with tabs[1]:
        st.header("Data Explorer")

        # Create two columns for the load data section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Load Data from Files")
            folder_path = st.text_input("Propaq JSON folder path", "/Users/rajveersingh/Downloads/propaq_data2/")
            
            # Add a file upload section
            st.write("---")
            st.subheader("Or Upload JSON Files")
            uploaded_files = st.file_uploader("Upload Propaq JSON files", type=["json"], accept_multiple_files=True)
            
            if uploaded_files:
                if st.button("Process Uploaded Files"):
                    with st.spinner("Processing uploaded files..."):
                        # Create a temporary directory for the uploaded files
                        temp_dir = Path("temp_uploads")
                        temp_dir.mkdir(exist_ok=True)
                        
                        try:
                            # Save uploaded files to temp directory
                            for uploaded_file in uploaded_files:
                                file_path = temp_dir / uploaded_file.name
                                with open(file_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                            
                            # Process the files
                            df = load_and_sort_data(str(temp_dir))
                            
                            if df.empty:
                                st.error("No valid data found in uploaded files.")
                            else:
                                st.session_state["df"] = df
                                st.session_state["data_source"] = "uploaded_files"
                                st.success(f"Loaded {len(df)} rows from {len(uploaded_files)} uploaded files!")
                                
                                # Automatically save to database
                                dataset_id = save_to_database(
                                    df, 
                                    name=f"Uploaded Files {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                                    description=f"Data from {len(uploaded_files)} uploaded files",
                                    source="File Upload"
                                )
                                
                                if dataset_id:
                                    st.info("Data automatically saved to database for future access.")
                        finally:
                            # Clean up temp directory
                            shutil.rmtree(temp_dir, ignore_errors=True)
            
            if st.button("Load Data from Files"):
                with st.spinner("Loading & processing data..."):
                    df = load_and_sort_data(folder_path)
                if df.empty:
                    st.error("No data found or parse error.")
                else:
                    st.session_state["df"] = df
                    st.session_state["data_source"] = "files"
                    st.session_state["folder_path"] = folder_path
                    st.success(f"Loaded {len(df)} rows from {folder_path}!")
                    
                    # Add option to save to database
                    if st.button("Save to Database"):
                        with st.spinner("Saving data to database..."):
                            dataset_id = save_to_database(df, 
                                                         name=f"Data from {os.path.basename(folder_path)}", 
                                                         description="Loaded from local files", 
                                                         source=folder_path)
                            if dataset_id:
                                st.success("Data successfully saved to database!")
                            else:
                                st.error("Failed to save data to database.")
        
        with col2:
            st.subheader("Load Data from Database")
            
            # Get available datasets
            datasets = get_database_datasets()
            
            if datasets.empty:
                st.info("No datasets available in the database.")
            else:
                # Format dataset info for selectbox
                dataset_options = [f"{row['name']} - {row['upload_date_formatted']} ({row['num_records']} records)" 
                                   for _, row in datasets.iterrows()]
                dataset_ids = datasets['id'].tolist()
                
                selected_index = st.selectbox(
                    "Select a dataset to load", 
                    range(len(dataset_options)),
                    format_func=lambda i: dataset_options[i]
                )
                
                if st.button("Load from Database"):
                    with st.spinner("Loading data from database..."):
                        selected_dataset_id = dataset_ids[selected_index]
                        df = load_from_database(selected_dataset_id)
                    
                    if df.empty:
                        st.error("Failed to load data from database.")
                    else:
                        st.session_state["df"] = df
                        st.session_state["data_source"] = "database"
                        st.session_state["dataset_id"] = selected_dataset_id
                        st.success(f"Loaded {len(df)} rows from database!")
                
                # Option to delete dataset
                if st.button("Delete Selected Dataset"):
                    with st.spinner("Deleting dataset..."):
                        selected_dataset_id = dataset_ids[selected_index]
                        delete_database_dataset(selected_dataset_id)
                        st.success("Dataset deleted successfully!")
                        st.experimental_rerun()

        # Attempt to auto-load data from database if no data is loaded
        if "df" not in st.session_state:
            # Check if there's data in the database
            datasets = get_database_datasets()
            if not datasets.empty:
                with st.spinner("Auto-loading most recent data from database..."):
                    df = load_from_database()  # Loads the most recent dataset
                if not df.empty:
                    st.session_state["df"] = df
                    st.session_state["data_source"] = "database"
                    st.info("Automatically loaded most recent data from database. Use the options above to load different data.")

        if "df" in st.session_state:
            df = st.session_state["df"]
            
            # Display data source info
            if st.session_state.get("data_source") == "files":
                st.info(f"Data source: Local files from {st.session_state.get('folder_path', 'unknown path')}")
            elif st.session_state.get("data_source") == "database":
                dataset_id = st.session_state.get("dataset_id")
                dataset_info = datasets[datasets['id'] == dataset_id].iloc[0] if dataset_id and not datasets.empty else None
                if dataset_info is not None:
                    st.info(f"Data source: Database - {dataset_info['name']} (Uploaded: {dataset_info['upload_date_formatted']})")
            
            # Data exploration section
            st.subheader("Explore Data")
            
            # Always filter to only show data within simulation windows
            df = df[df["InSimWindow"] == True]

            # 1) Filter by Sim
            sim_options = sorted([x for x in df["Sim"].dropna().unique() if x])
            sim_choice = st.selectbox("Filter by Simulation (optional)", ["<all>"] + sim_options)

            tmp_df = df if sim_choice=="<all>" else df[df["Sim"]==sim_choice]

            # 2) Filter by Date
            date_options = sorted([x for x in tmp_df["DateStr"].dropna().unique() if x])
            date_choice = st.selectbox("Filter by Date (optional)", ["<all>"] + date_options)

            if date_choice=="<all>":
                tmp_df2 = tmp_df
            else:
                tmp_df2 = tmp_df[tmp_df["DateStr"]==date_choice]

            # 3) Filter by Mannequin - only show valid mannequins for selected sim
            if sim_choice != "<all>":
                valid_mannequins = SIM_MANNEQUINS.get(sim_choice, [])
                man_options = sorted([x for x in valid_mannequins])
            else:
                man_options = sorted([x for x in tmp_df2["overrideMannequin"].dropna().unique() if x])
            
            man_choice = st.selectbox("Filter by Mannequin (optional)", ["<all>"] + man_options)

            # 4) Start/end time
            st.write("Optionally specify Start/End times (YYYY-MM-DD HH:MM) for narrower filtering:")
            start_time_str = st.text_input("Start Time", "")
            end_time_str = st.text_input("End Time", "")

            # Apply filters
            filtered_df = df.copy()
            if sim_choice!="<all>":
                filtered_df = filtered_df[filtered_df["Sim"]==sim_choice]
            if date_choice!="<all>":
                filtered_df = filtered_df[filtered_df["DateStr"]==date_choice]
            if man_choice!="<all>":
                filtered_df = filtered_df[filtered_df["overrideMannequin"]==man_choice]

            if start_time_str.strip():
                try:
                    stt = parser.parse(start_time_str)
                    filtered_df = filtered_df[filtered_df["TimeObj"] >= stt]
                except:
                    st.warning("Could not parse Start Time. Ignoring it.")
            if end_time_str.strip():
                try:
                    ett = parser.parse(end_time_str)
                    filtered_df = filtered_df[filtered_df["TimeObj"] <= ett]
                except:
                    st.warning("Could not parse End Time. Ignoring it.")

            st.write(f"**Filtered Rows**: {len(filtered_df)}")
            st.dataframe(filtered_df.head(50))

            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
            if not numeric_cols.empty:
                st.write("**Summary Stats**:")
                st.write(filtered_df[numeric_cols].describe())
            else:
                st.write("*No numeric columns found in this filtered set.*")
                
            # Option to save filtered data
            if st.button("Save Filtered Data to Database"):
                with st.spinner("Saving filtered data to database..."):
                    description = f"Filtered data - Sim: {sim_choice}, Date: {date_choice}, Mannequin: {man_choice}"
                    if start_time_str or end_time_str:
                        description += f" - Time range: {start_time_str} to {end_time_str}"
                    
                    dataset_id = save_to_database(
                        filtered_df, 
                        name=f"Filtered Data {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        description=description,
                        source=f"Filtered from: {st.session_state.get('data_source', 'unknown')}"
                    )
                    
                    if dataset_id:
                        st.success("Filtered data successfully saved to database!")
                    else:
                        st.error("Failed to save filtered data to database.")

    ###########################################################################
    # TAB 2: CPG COMPLIANCE
    ###########################################################################
    with tabs[2]:
        st.header("CPG Compliance")

        if "df" not in st.session_state:
            st.warning("No data loaded (use Data Explorer).")
        else:
            df = st.session_state["df"]
            valid_df = df[df["IsValid"]==True].copy()
            if valid_df.empty:
                st.info("No valid data within sim windows.")
            else:
                # Add debug mode toggle
                debug_mode = st.checkbox("Enable Debug Mode", value=False, help="Show additional diagnostic information")
                
                group_cols = ["Course","Sim","overrideMannequin","DateStr","Scenario"]
                unique_groups = valid_df[group_cols].drop_duplicates()
                if unique_groups.empty:
                    st.write("No group data found.")
                else:
                    group_options = unique_groups.apply(
                        lambda x: f"{x['Course']} | {x['Sim']} | {x['overrideMannequin']} | {x['DateStr']} | {x['Scenario']}",
                        axis=1
                    )
                    pick_group = st.selectbox("Pick a group for compliance", group_options)

                    # scenario override
                    all_scenarios = sorted(list(CPG_THRESHOLDS.keys()))
                    scenario_override = st.selectbox("Override scenario?", ["<none>"] + all_scenarios)

                    if st.button("Compute CPG Compliance"):
                        co, si, man, ds, sc = [x.strip() for x in pick_group.split("|")]
                        sub = valid_df[
                            (valid_df["Course"]==co)&
                            (valid_df["Sim"]==si)&
                            (valid_df["overrideMannequin"]==man)&
                            (valid_df["DateStr"]==ds)
                        ].copy()

                        # Pass original scenario when overriding
                        final_scenario = sc if scenario_override=="<none>" else scenario_override
                        
                        # Check and fix vital sign data
                        fixed_sub, fix_messages = check_and_fix_vital_data(sub, final_scenario)
                        
                        # Show data fix messages if in debug mode
                        if debug_mode and fix_messages:
                            st.write("### Data Preprocessing")
                            for msg in fix_messages:
                                if msg.startswith("ERROR"):
                                    st.error(msg)
                                else:
                                    st.info(f"🔧 {msg}")
                        
                        # Compute compliance with fixed data
                        summary = compute_time_out_of_range(fixed_sub, final_scenario, sc if scenario_override != "<none>" else None)
                        
                        # Display summary in a more organized way
                        st.write("### Simulation Compliance Report")
                        st.write(f"**Scenario**: {summary['scenario']}")
                        if summary.get('original_scenario'):
                            st.write(f"**Original Scenario**: {summary['original_scenario']}")
                        st.write(f"**Duration**: {summary['total_duration_min']} minutes")
                        st.write(f"**Overall Compliance Score**: {summary['compliance_score']}%")
                        
                        # Display color-coded assessment
                        assessment_color = {
                            "EXCELLENT": "success",
                            "GOOD": "info",
                            "FAIR": "warning",
                            "POOR": "error"
                        }
                        
                        # Check if assessment key exists in compliance_result
                        if 'assessment' in summary:
                            status = summary['assessment'].split(' - ')[0]
                            st.markdown(f"**Overall Assessment**: :{assessment_color[status]}[{summary['assessment']}]")
                        else:
                            st.markdown("**Overall Assessment**: :error[Unable to determine assessment - insufficient data]")
                        
                        # Display events first, as they may contain important notes about data quality
                        if summary['events']:
                            st.write("\n### Events and Notifications")
                            for event in summary['events']:
                                # Filter debug messages based on debug mode
                                if event.startswith("DEBUG:") and not debug_mode:
                                    continue
                                    
                                if event.startswith("CRITICAL"):
                                    st.error(event)
                                elif event.startswith("WARNING"):
                                    st.warning(event)
                                elif event.startswith("DEBUG:"):
                                    st.info(f"🔍 {event}")
                                else:
                                    st.info(event)
                        
                        # Display vital signs details in a table
                        st.write("\n### Vital Signs Details")
                        vital_rows = []
                        for vital, stats in summary['vital_stats'].items():
                            vital_rows.append({
                                "Vital Sign": vital,
                                "Target Range": stats['target'],
                                "Actual Range": stats['range'],
                                "Mean Value": stats['mean_value'],
                                "Compliance": f"{stats['in_range_percent']}%",
                                "Coverage": f"{stats['coverage_percent']}%",
                                "Status": stats['status'],
                                "Minutes Out of Range": stats['time_out_of_range_min'],
                                "Minutes Measured": stats['time_measured_min']
                            })
                        
                        if vital_rows:
                            display_vital_stats_table(vital_rows)
                        
                        # Add export option with more detailed information
                        export_data = {
                            "Report_Summary": {
                                "Scenario": summary['scenario'],
                                "Original_Scenario": summary.get('original_scenario', "N/A"),
                                "Duration_Minutes": summary['total_duration_min'],
                                "Overall_Compliance": summary['compliance_score'],
                                "Assessment": summary.get('assessment', "Unable to determine assessment - insufficient data")
                            },
                            "Events": summary['events'],
                            "Vital_Stats": summary['vital_stats']
                        }
                        
                        # Convert to CSV-friendly format
                        csv_buf = io.StringIO()
                        pd.json_normalize(export_data, sep='_').to_csv(csv_buf, index=False)
                        st.download_button(
                            "Download Full Report (CSV)",
                            data=csv_buf.getvalue(),
                            file_name="compliance_report.csv",
                            mime="text/csv"
                        )

    ###########################################################################
    # TAB 3: VISUALIZATION
    ###########################################################################
    with tabs[3]:
        st.header("Visualization")

        if "df" not in st.session_state:
            st.warning("No data loaded yet.")
        else:
            df = st.session_state["df"]
            valid_df = df[df["IsValid"]==True].copy()
            if valid_df.empty:
                st.info("No valid data for charting.")
            else:
                # Create two columns for layout
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    group_cols = ["Course","Sim","overrideMannequin","DateStr","Scenario"]
                    unique_groups = valid_df[group_cols].drop_duplicates()
                    if unique_groups.empty:
                        st.write("No group data found.")
                    else:
                        group_opts = unique_groups.apply(
                            lambda x: f"{x['Course']} | {x['Sim']} | {x['overrideMannequin']} | {x['DateStr']} | {x['Scenario']}",
                            axis=1
                        )
                        selected_group = st.selectbox("Select group to plot", group_opts)
                        
                        # Enhanced visualization options
                        viz_type = st.radio("Visualization Type", 
                            ["Basic Charts", "Advanced Analytics", "Intervention Analysis"], 
                            horizontal=True
                        )
                        
                        if viz_type == "Basic Charts":
                            chart_type = st.radio("Chart Type", 
                                ["Line", "Area", "Bar"], 
                                horizontal=True
                            )
                
                with col2:
                    # Analysis options
                    st.subheader("Analysis Options")
                    
                    # Select vitals to analyze
                    all_cols = [c for c in df.columns if c not in [
                        "TimeObj","TimeStr","DevSerial","SourceFile","rawMannequin","Course",
                        "Sim","overrideMannequin","DateStr","InSimWindow","IsValid","Scenario"
                    ] and not c.endswith("_status")]
                    
                    # Define default vitals but filter to only include those available in the dataset
                    preferred_vitals = ["NIBP_MAP", "Hr", "SpO2", "Temp1", "EtCO2"]
                    default_vitals = [v for v in preferred_vitals if v in all_cols]
                    
                    # If no preferred vitals are available, use the first 3 available columns (if any)
                    if not default_vitals and all_cols:
                        default_vitals = all_cols[:min(3, len(all_cols))]
                        
                    chosen_cols = st.multiselect("Select Vitals", all_cols, default=default_vitals)
                    
                    # Analysis parameters
                    if viz_type in ["Advanced Analytics", "Intervention Analysis"]:
                        window_size = st.slider("Analysis Window (minutes)", 
                            min_value=1, max_value=15, value=5
                        )
                        
                        show_options = st.multiselect("Show Additional Analysis",
                            ["Rolling Average", "Trend Lines", "Event Markers", "Variability Bands"],
                            default=["Rolling Average", "Event Markers"]
                        )
                        
                        if viz_type == "Intervention Analysis":
                            st.write("### Prediction Settings")
                            prediction_threshold = st.slider(
                                "Prediction Confidence Threshold",
                                min_value=0.0,
                                max_value=1.0,
                                value=0.7,
                                step=0.1
                            )

                if st.button("Generate Visualization"):
                    co, si, man, ds, sc = [x.strip() for x in selected_group.split("|")]
                    sub = valid_df[
                        (valid_df["Course"]==co)&
                        (valid_df["Sim"]==si)&
                        (valid_df["overrideMannequin"]==man)&
                        (valid_df["DateStr"]==ds)
                    ].copy()
                    
                    if sub.empty:
                        st.error("No rows found for that selection.")
                    else:
                        sub["ElapsedMin"] = (sub["TimeObj"] - sub["TimeObj"].iloc[0]).dt.total_seconds()/60.0
                        
                        if not chosen_cols:
                            st.write("No columns chosen to plot.")
                        else:
                            # Convert columns to numeric
                            for c in chosen_cols:
                                sub[c] = pd.to_numeric(sub[c], errors="coerce")
                            
                            if viz_type == "Basic Charts":
                                # Basic charting
                                if chart_type=="Line":
                                    st.line_chart(data=sub, x="ElapsedMin", y=chosen_cols)
                                elif chart_type=="Area":
                                    st.area_chart(sub.set_index("ElapsedMin")[chosen_cols])
                                else:
                                    st.bar_chart(sub.set_index("ElapsedMin")[chosen_cols])
                            else:
                                # Advanced analytics visualization
                                st.subheader("Advanced Analysis")
                                
                                # Detect clinical events
                                events = detect_clinical_events(sub, sc)
                                
                                if viz_type == "Intervention Analysis":
                                    # Generate and filter predictions
                                    predictions = predict_interventions(sub, chosen_cols)
                                    predictions = [
                                        p for p in predictions 
                                        if p['confidence'] >= prediction_threshold
                                    ]
                                    
                                    # Display intervention analysis
                                    display_intervention_analysis(sub, events, predictions)
                                else:
                                    # Create tabs for different analysis views
                                    analysis_tabs = st.tabs(["Vital Signs", "Events", "Trends"])
                                    
                                    with analysis_tabs[0]:
                                        for vital in chosen_cols:
                                            st.write(f"### {vital}")
                                            
                                            # Get trend analysis
                                            trend_analysis = analyze_trends(sub, vital, window_size)
                                            
                                            # Display trend information
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("Mean", f"{trend_analysis['mean']:.1f}")
                                            with col2:
                                                st.metric("Trend", trend_analysis['trend'].title())
                                            with col3:
                                                stability = "Stable" if trend_analysis['stability'] < 0.2 else "Variable"
                                                st.metric("Stability", stability)
                                            
                                            # Plot the vital sign with analysis overlay
                                            fig = create_advanced_vital_plot(sub, vital, window_size, show_options)
                                            st.plotly_chart(fig, use_container_width=True)
                                    
                                    with analysis_tabs[1]:
                                        if events:
                                            st.write("### Detected Events")
                                            events_df = pd.DataFrame(events)
                                            events_df['time'] = events_df['timestamp'].dt.strftime('%H:%M:%S')
                                            st.dataframe(
                                                events_df[['time', 'vital', 'type', 'description']]
                                            )
                                        else:
                                            st.info("No significant events detected")
                                    
                                    with analysis_tabs[2]:
                                        st.write("### Trend Analysis")
                                        trend_rows = []
                                        for vital in chosen_cols:
                                            analysis = analyze_trends(sub, vital, window_size)
                                            trend_rows.append({
                                                "Vital": vital,
                                                "Trend": analysis['trend'],
                                                "Stability Score": f"{analysis['stability']:.2f}",
                                                "Alerts": ", ".join(analysis['alerts'])
                                            })
                                        st.table(pd.DataFrame(trend_rows))
                                
                                # Show basic stats regardless of visualization type
                                with st.expander("Basic Statistics"):
                                    stats_rows = []
                                    for c in chosen_cols:
                                        stats_rows.append({
                                            "Vital": c,
                                            "Min": sub[c].min(),
                                            "Max": sub[c].max(),
                                            "Mean": sub[c].mean()
                                        })
                                    st.table(pd.DataFrame(stats_rows))

    ###########################################################################
    # TAB 4: REAL-TIME MONITOR
    ###########################################################################
    with tabs[4]:
        st.header("Real-Time Vital Signs Monitor")
        
        # Create a modern UI layout
        st.markdown("""
        <style>
        .monitor-container {
            background-color: #1E1E1E;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .monitor-header {
            color: #FFFFFF;
            font-size: 1.2rem;
            margin-bottom: 15px;
            display: flex;
            justify-content: space-between;
        }
        .config-container {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 15px;
            background-color: #f8f9fa;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Initialize session state for real-time monitoring
        if "streaming_active" not in st.session_state:
            st.session_state.streaming_active = False
        if "streaming_data" not in st.session_state:
            st.session_state.streaming_data = pd.DataFrame()
        if "last_update_time" not in st.session_state:
            st.session_state.last_update_time = datetime.now()
        if "data_source" not in st.session_state:
            st.session_state.data_source = "simulation"
        if "connection_status" not in st.session_state:
            st.session_state.connection_status = "Disconnected"
        if "last_successful_poll" not in st.session_state:
            st.session_state.last_successful_poll = None
        
        # Control panel
        with st.expander("Monitor Controls", expanded=True):
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                data_source = st.radio(
                    "Data Source",
                    ["Simulation", "Zoll Monitor", "WebSocket", "File Import"],
                    index=0
                )
                st.session_state.data_source = data_source.lower().replace(" ", "_")
            
            with col2:
                if st.session_state.data_source == "simulation":
                    update_frequency = st.slider(
                        "Update Frequency (seconds)",
                        min_value=1,
                        max_value=10,
                        value=2
                    )
                    st.session_state.update_frequency = update_frequency
                elif st.session_state.data_source == "zoll_monitor":
                    st.info("Configure Zoll Monitor connection in the Network Settings below")
                    polling_interval = st.slider(
                        "Polling Interval (seconds)",
                        min_value=1,
                        max_value=30,
                        value=5
                    )
                    st.session_state.polling_interval = polling_interval
                elif st.session_state.data_source == "websocket":
                    websocket_url = st.text_input(
                        "WebSocket URL",
                        value="ws://localhost:8765"
                    )
                    st.session_state.websocket_url = websocket_url
                elif st.session_state.data_source == "file_import":
                    uploaded_file = st.file_uploader(
                        "Upload JSON File",
                        type=["json"]
                    )
                    st.session_state.uploaded_file = uploaded_file
            
            with col3:
                all_scenarios = sorted(list(CPG_THRESHOLDS.keys()))
                selected_scenario = st.selectbox(
                    "Scenario for Compliance",
                    all_scenarios,
                    index=0
                )
                st.session_state.selected_scenario = selected_scenario
            
            # Control buttons
            start_col, stop_col, clear_col = st.columns([1, 1, 1])
            
            with start_col:
                if not st.session_state.streaming_active:
                    if st.button("Start Streaming", key="start_streaming"):
                        st.session_state.streaming_active = True
                        st.session_state.last_update_time = datetime.now()
                        st.experimental_rerun()
            
            with stop_col:
                if st.session_state.streaming_active:
                    if st.button("Stop Streaming", key="stop_streaming"):
                        st.session_state.streaming_active = False
                        st.experimental_rerun()
            
            with clear_col:
                if st.button("Clear Data", key="clear_data"):
                    st.session_state.streaming_data = pd.DataFrame()
                    st.session_state.last_update_time = datetime.now()
                    st.experimental_rerun()
        
        # Network Configuration Panel (for Zoll Monitor)
        if st.session_state.data_source == "zoll_monitor":
            with st.expander("Network Settings", expanded=True):
                st.markdown('<div class="config-container">', unsafe_allow_html=True)
                
                # Connection settings
                st.subheader("Zoll Monitor Connection")
                
                conn_col1, conn_col2 = st.columns(2)
                
                with conn_col1:
                    monitor_ip = st.text_input(
                        "Monitor IP Address",
                        value="192.168.1.100",
                        help="IP address of the Zoll Propaq monitor"
                    )
                    st.session_state.monitor_ip = monitor_ip
                    
                    monitor_port = st.number_input(
                        "Port",
                        min_value=1,
                        max_value=65535,
                        value=80,
                        help="Port for HTTP connection (usually 80 or 3128)"
                    )
                    st.session_state.monitor_port = monitor_port
                
                with conn_col2:
                    connection_method = st.selectbox(
                        "Connection Method",
                        ["HTTP Polling", "TCP Direct", "UDP Stream", "File Export"],
                        index=0,
                        help="Method to retrieve data from the monitor"
                    )
                    st.session_state.connection_method = connection_method
                    
                    use_proxy = st.checkbox(
                        "Use Proxy Server",
                        value=False,
                        help="Use a proxy server to handle authentication"
                    )
                    st.session_state.use_proxy = use_proxy
                
                # Authentication settings
                st.subheader("Authentication")
                
                auth_col1, auth_col2 = st.columns(2)
                
                with auth_col1:
                    auth_required = st.checkbox(
                        "Authentication Required",
                        value=True,
                        help="Enable if the monitor requires authentication"
                    )
                    st.session_state.auth_required = auth_required
                    
                    if auth_required:
                        auth_username = st.text_input(
                            "Username",
                            value="admin",
                            help="Username for monitor authentication"
                        )
                        st.session_state.auth_username = auth_username
                
                with auth_col2:
                    if auth_required:
                        auth_password = st.text_input(
                            "Password",
                            value="",
                            type="password",
                            help="Password for monitor authentication"
                        )
                        st.session_state.auth_password = auth_password
                        
                        auth_method = st.selectbox(
                            "Authentication Method",
                            ["Basic", "Digest", "Token"],
                            index=0,
                            help="Authentication method required by the monitor"
                        )
                        st.session_state.auth_method = auth_method
                
                # Network configuration
                st.subheader("Network Configuration")
                
                net_col1, net_col2 = st.columns(2)
                
                with net_col1:
                    bypass_captive = st.checkbox(
                        "Bypass Captive Portal",
                        value=False,
                        help="Attempt to bypass network captive portal"
                    )
                    st.session_state.bypass_captive = bypass_captive
                    
                    if bypass_captive:
                        mac_whitelist = st.text_input(
                            "MAC Address Whitelist",
                            value="",
                            help="Comma-separated list of whitelisted MAC addresses"
                        )
                        st.session_state.mac_whitelist = mac_whitelist
                
                with net_col2:
                    use_dedicated_vlan = st.checkbox(
                        "Use Dedicated VLAN",
                        value=False,
                        help="Connect through a dedicated VLAN for medical devices"
                    )
                    st.session_state.use_dedicated_vlan = use_dedicated_vlan
                    
                    if use_dedicated_vlan:
                        vlan_id = st.number_input(
                            "VLAN ID",
                            min_value=1,
                            max_value=4094,
                            value=100,
                            help="ID of the dedicated VLAN"
                        )
                        st.session_state.vlan_id = vlan_id
                
                # Advanced settings
                with st.expander("Advanced Settings"):
                    timeout = st.slider(
                        "Connection Timeout (seconds)",
                        min_value=1,
                        max_value=60,
                        value=10,
                        help="Maximum time to wait for a response from the monitor"
                    )
                    st.session_state.timeout = timeout
                    
                    retry_attempts = st.slider(
                        "Retry Attempts",
                        min_value=1,
                        max_value=10,
                        value=3,
                        help="Number of retry attempts if connection fails"
                    )
                    st.session_state.retry_attempts = retry_attempts
                    
                    data_format = st.selectbox(
                        "Data Format",
                        ["JSON", "XML", "CSV", "Binary"],
                        index=0,
                        help="Expected format of the data from the monitor"
                    )
                    st.session_state.data_format = data_format
                    
                    custom_headers = st.text_area(
                        "Custom HTTP Headers",
                        value="User-Agent: VitalsApp/1.5\nAccept: application/json",
                        help="Custom HTTP headers for requests (one per line)"
                    )
                    st.session_state.custom_headers = custom_headers
                
                # Test connection button
                if st.button("Test Connection"):
                    with st.spinner("Testing connection to Zoll monitor..."):
                        # Create a client
                        client = ZollMonitorClient(
                            ip_address=st.session_state.monitor_ip,
                            port=st.session_state.monitor_port,
                            connection_method=st.session_state.connection_method,
                            auth_required=st.session_state.auth_required,
                            username=st.session_state.auth_username if st.session_state.auth_required else "",
                            password=st.session_state.auth_password if st.session_state.auth_required else "",
                            auth_method=st.session_state.auth_method if st.session_state.auth_required else "Basic",
                            use_proxy=st.session_state.use_proxy,
                            timeout=st.session_state.timeout,
                            retry_attempts=st.session_state.retry_attempts
                        )
                        
                        # Test the connection
                        success, message = client.test_connection()
                        
                        if success:
                            st.success(f"Successfully connected to monitor at {st.session_state.monitor_ip}:{st.session_state.monitor_port}")
                            st.session_state.connection_status = "Connected"
                        else:
                            st.error(f"Failed to connect: {message}")
                            st.session_state.connection_status = "Disconnected"
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Display the monitor UI
        st.markdown('<div class="monitor-container">', unsafe_allow_html=True)
        st.markdown('<div class="monitor-header"><span>Patient Monitor</span><span>Scenario: ' + selected_scenario + '</span></div>', unsafe_allow_html=True)
        
        # Update streaming data if active
        if st.session_state.streaming_active:
            update_streaming_data()
        
        # Get the most recent data
        if not st.session_state.streaming_data.empty:
            latest_data = st.session_state.streaming_data.iloc[-1].to_dict()
        else:
            # Use sample data if no streaming data is available
            latest_data = {
                "Hr": 75,
                "SpO2": 98,
                "NIBP_SYS": 120,
                "NIBP_DIA": 80,
                "NIBP_MAP": 93,
                "Temp1": 98.6,
                "RespRate": 16,
                "EtCO2": 35
            }
        
        # Create a grid layout for vital signs
        col1, col2, col3, col4 = st.columns(4)
        
        # Display vital signs in the grid
        with col1:
            st.markdown(
                display_vital_sign_box(
                    "Heart Rate", 
                    int(latest_data.get("Hr", 75)), 
                    "bpm", 
                    latest_data.get("Hr_status", "valid"),
                    normal_range=(60, 100),
                    thresholds=CPG_THRESHOLDS.get(selected_scenario, {}).get("Hr")
                ),
                unsafe_allow_html=True
            )
            
            st.markdown(
                display_vital_sign_box(
                    "Temperature", 
                    round(latest_data.get("Temp1", 98.6), 1), 
                    "°F", 
                    latest_data.get("Temp1_status", "valid"),
                    normal_range=(97, 99),
                    thresholds=CPG_THRESHOLDS.get(selected_scenario, {}).get("TempF")
                ),
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                display_vital_sign_box(
                    "SpO2", 
                    int(latest_data.get("SpO2", 98)), 
                    "%", 
                    latest_data.get("SpO2_status", "valid"),
                    normal_range=(95, 100),
                    thresholds=CPG_THRESHOLDS.get(selected_scenario, {}).get("SpO2")
                ),
                unsafe_allow_html=True
            )
            
            st.markdown(
                display_vital_sign_box(
                    "Resp Rate", 
                    int(latest_data.get("RespRate", 16)), 
                    "bpm", 
                    latest_data.get("RespRate_status", "valid"),
                    normal_range=(12, 20),
                    thresholds=CPG_THRESHOLDS.get(selected_scenario, {}).get("RespRate")
                ),
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                display_vital_sign_box(
                    "Blood Pressure", 
                    f"{int(latest_data.get('NIBP_SYS', 120))}/{int(latest_data.get('NIBP_DIA', 80))}", 
                    "mmHg", 
                    latest_data.get("NIBP_SYS_status", "valid"),
                    normal_range=None,
                    thresholds=None
                ),
                unsafe_allow_html=True
            )
            
            st.markdown(
                display_vital_sign_box(
                    "EtCO2", 
                    int(latest_data.get("EtCO2", 35)), 
                    "mmHg", 
                    latest_data.get("EtCO2_status", "valid"),
                    normal_range=(35, 45),
                    thresholds=CPG_THRESHOLDS.get(selected_scenario, {}).get("EtCO2")
                ),
                unsafe_allow_html=True
            )
        
        with col4:
            st.markdown(
                display_vital_sign_box(
                    "MAP", 
                    int(latest_data.get("NIBP_MAP", 93)), 
                    "mmHg", 
                    latest_data.get("NIBP_MAP_status", "valid"),
                    normal_range=(70, 105),
                    thresholds=CPG_THRESHOLDS.get(selected_scenario, {}).get("NIBP_MAP")
                ),
                unsafe_allow_html=True
            )
            
            # Add connection status indicator
            if st.session_state.data_source == "zoll_monitor":
                # Determine status color
                status_color = "#00CC00" if st.session_state.connection_status == "Connected" else "#FF4B4B"
                background_color = "#EEFFEE" if st.session_state.connection_status == "Connected" else "#FFEEEE"
                
                # Format last update time
                last_update = "Never"
                if st.session_state.last_successful_poll:
                    time_diff = datetime.now() - st.session_state.last_successful_poll
                    if time_diff.total_seconds() < 60:
                        last_update = f"{int(time_diff.total_seconds())} seconds ago"
                    else:
                        last_update = f"{int(time_diff.total_seconds() / 60)} minutes ago"
                
                st.markdown(f"""
                <div style="
                    background-color: {background_color};
                    border-radius: 10px;
                    padding: 10px;
                    margin: 5px 0;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                ">
                    <h3 style="margin: 0; font-size: 1rem; color: #555;">Connection Status</h3>
                    <div style="display: flex; align-items: center; margin-top: 5px;">
                        <div style="
                            width: 12px;
                            height: 12px;
                            border-radius: 50%;
                            background-color: {status_color};
                            margin-right: 8px;
                        "></div>
                        <span style="color: {status_color}; font-weight: bold;">{st.session_state.connection_status}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #777; margin-top: 5px;">
                        Last update: {last_update}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display trend charts
        st.markdown("### Real-Time Trends")
        
        # Create tabs for different vital sign groups
        trend_tabs = st.tabs(["Cardiovascular", "Respiratory", "Temperature"])
        
        with trend_tabs[0]:
            # Cardiovascular vital signs
            cv_vitals = ["Hr", "NIBP_SYS", "NIBP_DIA", "NIBP_MAP"]
            cv_vitals = [v for v in cv_vitals if v in st.session_state.streaming_data.columns]
            
            if cv_vitals:
                # Convert columns to numeric
                for v in cv_vitals:
                    st.session_state.streaming_data[v] = pd.to_numeric(st.session_state.streaming_data[v], errors="coerce")
                
                # Create the plot
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add heart rate
                if "Hr" in cv_vitals:
                    fig.add_trace(
                        plotly_go.Scatter(
                            x=st.session_state.streaming_data["ElapsedMin"],
                            y=st.session_state.streaming_data["Hr"],
                            name="Heart Rate",
                            line=dict(color="red", width=2)
                        )
                    )
                
                # Add blood pressure
                if "NIBP_SYS" in cv_vitals and "NIBP_DIA" in cv_vitals:
                    fig.add_trace(
                        plotly_go.Scatter(
                            x=st.session_state.streaming_data["ElapsedMin"],
                            y=st.session_state.streaming_data["NIBP_SYS"],
                            name="Systolic BP",
                            line=dict(color="blue", width=2)
                        )
                    )
                    fig.add_trace(
                        plotly_go.Scatter(
                            x=st.session_state.streaming_data["ElapsedMin"],
                            y=st.session_state.streaming_data["NIBP_DIA"],
                            name="Diastolic BP",
                            line=dict(color="lightblue", width=2)
                        )
                    )
                
                # Add MAP
                if "NIBP_MAP" in cv_vitals:
                    fig.add_trace(
                        plotly_go.Scatter(
                            x=st.session_state.streaming_data["ElapsedMin"],
                            y=st.session_state.streaming_data["NIBP_MAP"],
                            name="MAP",
                            line=dict(color="purple", width=2)
                        )
                    )
                
                # Update layout
                fig.update_layout(
                    title="Cardiovascular Trends",
                    xaxis_title="Time (minutes)",
                    yaxis_title="Value",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with trend_tabs[1]:
            # Respiratory vital signs
            resp_vitals = ["SpO2", "RespRate", "EtCO2"]
            resp_vitals = [v for v in resp_vitals if v in st.session_state.streaming_data.columns]
            
            if resp_vitals:
                # Convert columns to numeric
                for v in resp_vitals:
                    st.session_state.streaming_data[v] = pd.to_numeric(st.session_state.streaming_data[v], errors="coerce")
                
                # Create the plot
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add SpO2
                if "SpO2" in resp_vitals:
                    fig.add_trace(
                        plotly_go.Scatter(
                            x=st.session_state.streaming_data["ElapsedMin"],
                            y=st.session_state.streaming_data["SpO2"],
                            name="SpO2",
                            line=dict(color="blue", width=2)
                        ),
                        secondary_y=False
                    )
                
                # Add respiratory rate
                if "RespRate" in resp_vitals:
                    fig.add_trace(
                        plotly_go.Scatter(
                            x=st.session_state.streaming_data["ElapsedMin"],
                            y=st.session_state.streaming_data["RespRate"],
                            name="Resp Rate",
                            line=dict(color="green", width=2)
                        ),
                        secondary_y=True
                    )
                
                # Add EtCO2
                if "EtCO2" in resp_vitals:
                    fig.add_trace(
                        plotly_go.Scatter(
                            x=st.session_state.streaming_data["ElapsedMin"],
                            y=st.session_state.streaming_data["EtCO2"],
                            name="EtCO2",
                            line=dict(color="orange", width=2)
                        ),
                        secondary_y=True
                    )
                
                # Update layout
                fig.update_layout(
                    title="Respiratory Trends",
                    xaxis_title="Time (minutes)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                
                # Update y-axis labels
                fig.update_yaxes(title_text="SpO2 (%)", secondary_y=False)
                fig.update_yaxes(title_text="Rate/EtCO2", secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
        
        with trend_tabs[2]:
            # Temperature
            temp_vitals = ["Temp1"]
            temp_vitals = [v for v in temp_vitals if v in st.session_state.streaming_data.columns]
            
            if temp_vitals:
                # Convert columns to numeric
                for v in temp_vitals:
                    st.session_state.streaming_data[v] = pd.to_numeric(st.session_state.streaming_data[v], errors="coerce")
                
                # Create the plot
                fig = plotly_go.Figure()
                
                # Add temperature
                if "Temp1" in temp_vitals:
                    fig.add_trace(
                        plotly_go.Scatter(
                            x=st.session_state.streaming_data["ElapsedMin"],
                            y=st.session_state.streaming_data["Temp1"],
                            name="Temperature",
                            line=dict(color="red", width=2)
                        )
                    )
                
                # Update layout
                fig.update_layout(
                    title="Temperature Trend",
                    xaxis_title="Time (minutes)",
                    yaxis_title="Temperature (°F)",
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Real-time CPG compliance
        st.markdown("### Real-Time CPG Compliance")
        
        # Add device and scenario information
        st.session_state.streaming_data["DevSerial"] = "SIMULATOR"
        st.session_state.streaming_data["Scenario"] = selected_scenario
        st.session_state.streaming_data["IsValid"] = True
        st.session_state.streaming_data["InSimWindow"] = True
        
        # Compute compliance
        compliance_result = compute_time_out_of_range(st.session_state.streaming_data, selected_scenario)
        
        # Display compliance score
        st.write(f"**Overall Compliance Score**: {compliance_result['compliance_score']}%")
        
        # Display color-coded assessment
        assessment_color = {
            "EXCELLENT": "success",
            "GOOD": "info",
            "FAIR": "warning",
            "POOR": "error"
        }
        
        # Check if assessment key exists in compliance_result
        if 'assessment' in compliance_result:
            status = compliance_result['assessment'].split(' - ')[0]
            st.markdown(f"**Assessment**: :{assessment_color[status]}[{compliance_result['assessment']}]")
        else:
            st.markdown("**Assessment**: :error[Unable to determine assessment - insufficient data]")
        
        # Display vital signs details in a table
        vital_rows = []
        for vital, stats in compliance_result['vital_stats'].items():
            vital_rows.append({
                "Vital Sign": vital,
                "Target Range": stats['target'],
                "Actual Range": stats['range'],
                "Mean Value": stats['mean_value'],
                "Compliance": f"{stats['in_range_percent']}%",
                "Status": stats['status']
            })
        
        if vital_rows:
            st.write("**Vital Signs Compliance**")
            st.dataframe(pd.DataFrame(vital_rows))
        
        # Option to save the current data
        if st.button("Save Current Session Data"):
            # Create a timestamp for the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vital_signs_session_{timestamp}.csv"
            
            # Convert the DataFrame to CSV
            csv = st.session_state.streaming_data.to_csv(index=False)
            
            # Provide a download button
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=filename,
                mime="text/csv"
            )

def create_advanced_vital_plot(df, vital, window_size, show_options):
    """Create an advanced plot using Plotly"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Check if vital exists in dataframe
    if vital not in df.columns:
        # Create empty figure with message
        fig = go.Figure()
        fig.update_layout(
            title=f"{vital} - Data Not Available",
            annotations=[dict(
                text="This vital sign is not available in the dataset",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )]
        )
        return fig
    
    # Convert to numeric and drop NaN values
    df_clean = df.copy()
    df_clean[vital] = pd.to_numeric(df_clean[vital], errors='coerce')
    
    # Check if we have any valid data after conversion
    if df_clean[vital].notna().sum() == 0:
        # Create empty figure with message
        fig = go.Figure()
        fig.update_layout(
            title=f"{vital} - No Valid Data",
            annotations=[dict(
                text="No valid measurements available for this vital sign",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )]
        )
        return fig
    
    # Create figure
    fig = go.Figure()
    
    # Add main vital sign line
    fig.add_trace(go.Scatter(
        x=df_clean['ElapsedMin'],
        y=df_clean[vital],
        name=vital,
        line=dict(color='#4361ee', width=2)
    ))
    
    if "Rolling Average" in show_options:
        # Add rolling average
        rolling_mean = df_clean[vital].rolling(window=window_size, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df_clean['ElapsedMin'],
            y=rolling_mean,
            name=f"{window_size}min Average",
            line=dict(color='#4cc9f0', width=2, dash='dash')
        ))
    
    if "Variability Bands" in show_options:
        # Add variability bands
        rolling_mean = df_clean[vital].rolling(window=window_size, min_periods=1).mean()
        rolling_std = df_clean[vital].rolling(window=window_size, min_periods=1).std()
        fig.add_trace(go.Scatter(
            x=df_clean['ElapsedMin'],
            y=rolling_mean + rolling_std,
            name='Upper Band',
            line=dict(color='#4cc9f0', width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=df_clean['ElapsedMin'],
            y=rolling_mean - rolling_std,
            name='Lower Band',
            fill='tonexty',
            fillcolor='rgba(76, 201, 240, 0.1)',
            line=dict(color='#4cc9f0', width=0),
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title=f"{vital} Over Time",
        xaxis_title="Elapsed Time (minutes)",
        yaxis_title=vital,
        hovermode='x unified',
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

# Update the CPG Compliance tab display code
def display_vital_stats_table(vital_rows):
    if not vital_rows:
        st.warning("No vital sign data available for analysis.")
        return
        
    vital_df = pd.DataFrame(vital_rows)
    vital_df = vital_df.rename(columns={
        "Vital Sign": "Vital Sign",
        "Target Range": "Target Range",
        "Actual Range": "Actual Range",
        "Mean Value": "Mean Value",
        "Compliance": "In-Range %",
        "Coverage": "Monitoring Coverage %",
        "Status": "Status",
        "Minutes Out of Range": "Minutes Out of Range",
        "Minutes Measured": "Minutes Measured"
    })
    
    # Apply conditional formatting
    def highlight_status(val):
        if val == "Excellent":
            return "background-color: #8eff8e"
        elif val == "Good":
            return "background-color: #b8ff8e"
        elif val == "Fair":
            return "background-color: #fff68e"
        elif val == "Poor":
            return "background-color: #ff8e8e"
        return ""
    
    # Display the table with styling
    st.dataframe(vital_df.style.applymap(highlight_status, subset=["Status"]))

def predict_interventions(df, vital_signs=['Hr', 'SpO2', 'NIBP_MAP', 'Temp1']):
    """Predict potential intervention points based on vital sign patterns"""
    predictions = []
    
    # Create a mapping of preferred vitals to potential alternatives
    vital_alternatives = {
        'NIBP_MAP': ['IBP1_MAP', 'IBP2_MAP', 'IBP3_MAP', 'MAP', 'MeanArterialPressure'],
        'NIBP_SYS': ['IBP1_SYS', 'IBP2_SYS', 'IBP3_SYS', 'SBP', 'Systolic'],
        'NIBP_DIA': ['IBP1_DIA', 'IBP2_DIA', 'IBP3_DIA', 'DBP', 'Diastolic'],
        'Hr': ['HR', 'HeartRate', 'Pulse'],
        'SpO2': ['SPO2', 'Spo2', 'O2Sat'],
        'Temp1': ['TempF', 'Temp', 'Temperature']
    }
    
    # Try to use the preferred vitals first, then fall back to alternatives if needed
    processed_vitals = []
    for vital in vital_signs:
        if vital in df.columns:
            processed_vitals.append(vital)
        else:
            # Try alternatives
            for alt in vital_alternatives.get(vital, []):
                if alt in df.columns:
                    processed_vitals.append(alt)
                    break
    
    # Filter to only include vitals that are actually available in the dataset
    valid_vitals = [v for v in processed_vitals if v in df.columns]
    
    # If no valid vitals are available, return empty predictions
    if not valid_vitals:
        return predictions
    
    # Prepare feature matrix
    features = []
    
    for vital in valid_vitals:
        series = pd.to_numeric(df[vital], errors='coerce')
        if series.notna().any():
            # Basic statistics
            features.extend([
                series,
                series.rolling(window=5, min_periods=1).mean(),
                series.rolling(window=5, min_periods=1).std(),
                series.diff() / df['ElapsedMin'].diff()  # rate of change
            ])
    
    if not features:
        return predictions
    
    # Combine features into matrix
    X = np.column_stack(features)
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Detect anomalies using Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomaly_scores = iso_forest.fit_predict(X_scaled)
    
    # Find potential intervention points
    for vital in valid_vitals:
        series = pd.to_numeric(df[vital], errors='coerce')
        if series.notna().any():
            # Detect significant changes
            peaks, _ = find_peaks(abs(series.diff()), height=series.std()*2)
            
            for peak in peaks:
                if anomaly_scores[peak] == -1:  # Anomaly detected
                    predictions.append({
                        'timestamp': df['TimeObj'].iloc[peak],
                        'vital': vital,
                        'type': 'predicted_intervention',
                        'confidence': calculate_confidence(series, peak),
                        'description': f"Potential intervention needed for {vital}"
                    })
    
    # Sort predictions by confidence
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    return predictions

def calculate_confidence(series, index, window=5):
    """Calculate confidence score for intervention prediction"""
    if index < window:
        window = index
    
    # Get window of data before the point
    window_data = series.iloc[index-window:index+1]
    
    # Calculate various factors
    volatility = window_data.std() / window_data.mean() if window_data.mean() != 0 else 1
    trend_strength = abs(window_data.iloc[-1] - window_data.iloc[0]) / window_data.std() if window_data.std() != 0 else 0
    persistence = sum(abs(window_data.diff()) > window_data.std()) / window
    
    # Combine factors into confidence score (0-1)
    confidence = (0.4 * volatility + 0.4 * trend_strength + 0.2 * persistence)
    return min(max(confidence, 0), 1)  # Clamp between 0 and 1

def analyze_intervention_timing(df, events, predictions):
    """Analyze timing of interventions relative to vital sign changes"""
    analysis = []
    
    # Group events by vital sign
    vital_events = {}
    for event in events:
        if event['vital'] not in vital_events:
            vital_events[event['vital']] = []
        vital_events[event['vital']].append(event)
    
    # Analyze each vital sign
    for vital, vital_predictions in groupby(predictions, key=lambda x: x['vital']):
        if vital not in vital_events:
            continue
            
        vital_predictions = list(vital_predictions)
        vital_series = pd.to_numeric(df[vital], errors='coerce')
        
        # Calculate lead time for each prediction
        for pred in vital_predictions:
            pred_time = pred['timestamp']
            
            # Find nearest actual event
            nearest_event = min(
                (e for e in vital_events[vital] if e['timestamp'] >= pred_time),
                key=lambda e: abs((e['timestamp'] - pred_time).total_seconds()),
                default=None
            )
            
            if nearest_event:
                lead_time = (nearest_event['timestamp'] - pred_time).total_seconds() / 60
                
                analysis.append({
                    'vital': vital,
                    'prediction_time': pred_time,
                    'event_time': nearest_event['timestamp'],
                    'lead_time_minutes': lead_time,
                    'confidence': pred['confidence'],
                    'accuracy': calculate_prediction_accuracy(
                        vital_series,
                        pred_time,
                        nearest_event['timestamp']
                    )
                })
    
    return analysis

def calculate_prediction_accuracy(series, pred_time, event_time):
    """Calculate accuracy of intervention prediction"""
    # Convert timestamps to indices
    pred_idx = series.index.get_loc(pred_time, method='nearest')
    event_idx = series.index.get_loc(event_time, method='nearest')
    
    # Get window of data between prediction and event
    window_data = series.iloc[pred_idx:event_idx+1]
    
    if window_data.empty:
        return 0.0
    
    # Calculate prediction accuracy based on:
    # 1. Trend direction accuracy
    # 2. Magnitude of change
    # 3. Consistency of change
    
    pred_value = window_data.iloc[0]
    event_value = window_data.iloc[-1]
    
    # Trend direction accuracy (0-0.4)
    trend_accuracy = 0.4 * (1 - min(1, abs(
        np.sign(event_value - pred_value) -
        np.sign(window_data.diff()).mode()[0]
    )))
    
    # Magnitude accuracy (0-0.4)
    expected_change = abs(event_value - pred_value)
    actual_change = abs(window_data.max() - window_data.min())
    magnitude_accuracy = 0.4 * (1 - min(1, abs(
        expected_change - actual_change
    ) / expected_change)) if expected_change != 0 else 0
    
    # Consistency (0-0.2)
    consistency = 0.2 * (1 - window_data.std() / window_data.mean()) if window_data.mean() != 0 else 0
    
    return trend_accuracy + magnitude_accuracy + consistency

# Update the visualization tab to include intervention predictions
def display_intervention_analysis(df, events, predictions):
    """Display intervention analysis in the visualization tab"""
    st.subheader("Intervention Analysis")
    
    # Show predictions
    if predictions:
        st.write("### Predicted Interventions")
        pred_df = pd.DataFrame(predictions)
        pred_df['time'] = pred_df['timestamp'].dt.strftime('%H:%M:%S')
        st.dataframe(
            pred_df[['time', 'vital', 'confidence', 'description']]
            .sort_values('confidence', ascending=False)
        )
        
        # Analyze timing
        timing_analysis = analyze_intervention_timing(df, events, predictions)
        if timing_analysis:
            st.write("### Intervention Timing Analysis")
            timing_df = pd.DataFrame(timing_analysis)
            timing_df['prediction_time'] = timing_df['prediction_time'].dt.strftime('%H:%M:%S')
            timing_df['event_time'] = timing_df['event_time'].dt.strftime('%H:%M:%S')
            
            st.dataframe(
                timing_df[[
                    'vital',
                    'prediction_time',
                    'event_time',
                    'lead_time_minutes',
                    'confidence',
                    'accuracy'
                ]]
            )
            
            # Show summary statistics
            st.write("### Summary Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_lead_time = timing_df['lead_time_minutes'].mean()
                st.metric(
                    "Average Lead Time",
                    f"{avg_lead_time:.1f} min"
                )
            
            with col2:
                avg_confidence = timing_df['confidence'].mean()
                st.metric(
                    "Average Confidence",
                    f"{avg_confidence:.1%}"
                )
            
            with col3:
                avg_accuracy = timing_df['accuracy'].mean()
                st.metric(
                    "Average Accuracy",
                    f"{avg_accuracy:.1%}"
                )
    else:
        st.info("No intervention predictions available")

###############################################################################
# REAL-TIME MONITORING FUNCTIONS
###############################################################################

def generate_simulated_vitals(previous_data=None):
    """
    Generate simulated vital signs data for testing the real-time monitor
    
    Args:
        previous_data: Optional previous data point to ensure realistic changes
        
    Returns:
        Dictionary with simulated vital signs
    """
    # Base values for vital signs (realistic ranges)
    base_values = {
        "Hr": 75,       # Heart rate: 60-100 bpm
        "SpO2": 98,     # Oxygen saturation: 95-100%
        "NIBP_SYS": 120, # Systolic BP: 90-140 mmHg
        "NIBP_DIA": 80,  # Diastolic BP: 60-90 mmHg
        "NIBP_MAP": 93,  # Mean arterial pressure: 70-105 mmHg
        "Temp1": 98.6,   # Temperature: 97-99°F
        "RespRate": 16,  # Respiratory rate: 12-20 breaths/min
        "EtCO2": 35      # End-tidal CO2: 35-45 mmHg
    }
    
    # Variation ranges for each vital sign
    variations = {
        "Hr": 3,        # +/- 3 bpm
        "SpO2": 1,      # +/- 1 %
        "NIBP_SYS": 5,  # +/- 5 mmHg
        "NIBP_DIA": 3,  # +/- 3 mmHg
        "NIBP_MAP": 3,  # +/- 3 mmHg
        "Temp1": 0.2,   # +/- 0.2 °F
        "RespRate": 1,  # +/- 1 breath/min
        "EtCO2": 2      # +/- 2 mmHg
    }
    
    # If we have previous data, use it as the base
    if previous_data is not None:
        for key in base_values.keys():
            if key in previous_data and previous_data[key] is not None:
                base_values[key] = previous_data[key]
    
    # Generate new values with random variations
    new_data = {}
    for key, base_value in base_values.items():
        variation = variations.get(key, 0)
        # Random variation within the specified range
        new_value = base_value + random.uniform(-variation, variation)
        # Ensure values stay within realistic ranges
        if key == "SpO2" and new_value > 100:
            new_value = 100
        elif key == "SpO2" and new_value < 90:
            new_value = 90 + random.uniform(0, 5)  # Keep SpO2 reasonable
        new_data[key] = round(new_value, 1)
    
    # Calculate MAP if it wasn't provided (approximation)
    if "NIBP_MAP" not in new_data and "NIBP_SYS" in new_data and "NIBP_DIA" in new_data:
        new_data["NIBP_MAP"] = round((new_data["NIBP_SYS"] + 2 * new_data["NIBP_DIA"]) / 3, 1)
    
    # Add timestamps and status
    new_data["TimeObj"] = datetime.now()
    new_data["TimeStr"] = new_data["TimeObj"].strftime("%Y-%m-%d %H:%M:%S")
    
    # Add status for each vital sign
    for key in base_values.keys():
        new_data[f"{key}_status"] = "valid"
    
    return new_data

def fetch_websocket_data(url):
    """Fetch data from a WebSocket server"""
    data = {
        "Hr": None,
        "SpO2": None,
        "NIBP_SYS": None,
        "NIBP_DIA": None,
        "NIBP_MAP": None,
        "Temp1": None,
        "RR": None,
        "EtCO2": None
    }
    
    try:
        # Create a WebSocket connection
        ws = create_connection(url, timeout=2)
        
        # Receive data
        result = ws.recv()
        ws.close()
        
        # Parse the data
        try:
            json_data = json.loads(result)
            for key in data.keys():
                if key in json_data:
                    data[key] = json_data[key]
        except json.JSONDecodeError:
            st.error("Failed to parse WebSocket data")
    except Exception as e:
        st.error(f"WebSocket connection error: {str(e)}")
    
    # Add timestamp
    data["TimeObj"] = datetime.now()
    data["TimeStr"] = data["TimeObj"].strftime("%Y-%m-%d %H:%M:%S")
    
    return data

def process_uploaded_file(uploaded_file):
    """
    Process an uploaded JSON file to extract vital signs data
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        DataFrame with vital signs data from the file
    """
    try:
        # Read the uploaded file
        data = json.load(uploaded_file)
        
        # Process the JSON data using existing functions
        if "ZOLL" in data and "FullDisclosure" in data["ZOLL"]:
            # Use the existing parsing function
            temp_file = "temp_upload.json"
            with open(temp_file, 'w') as f:
                json.dump(data, f)
            
            rows = parse_one_json_optimized(temp_file)
            os.remove(temp_file)
            
            if rows:
                return pd.DataFrame(rows)
        
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return pd.DataFrame()

def update_streaming_data():
    """Update streaming data based on selected data source"""
    if not st.session_state.streaming_active:
        return
    
    # Get current time
    current_time = datetime.now()
    
    # Check if we need to update based on data source and polling interval
    if st.session_state.data_source == "Simulation":
        # For simulation, check if enough time has passed since last update
        if 'last_update_time' in st.session_state and (current_time - st.session_state.last_update_time).total_seconds() < st.session_state.update_frequency:
            return
            
        # Generate new simulated data
        new_data = generate_simulated_vitals(st.session_state.streaming_data[-1] if st.session_state.streaming_data else None)
        
        # Add to streaming data
        st.session_state.streaming_data.append(new_data)
        st.session_state.last_update_time = current_time
        
    elif st.session_state.data_source == "Zoll Monitor":
        # For Zoll monitor, check if enough time has passed since last poll
        if 'last_update_time' in st.session_state and (current_time - st.session_state.last_update_time).total_seconds() < st.session_state.polling_interval:
            return
            
        # Use a small delay to prevent excessive CPU usage
        time_module.sleep(0.1)
            
        # Poll the Zoll monitor for vital signs
        config = {
            "ip_address": st.session_state.get("monitor_ip", "192.168.1.100"),
            "port": st.session_state.get("monitor_port", 80),
            "connection_method": st.session_state.get("connection_method", "HTTP Polling"),
            "auth_required": st.session_state.get("auth_required", False),
            "username": st.session_state.get("auth_username", ""),
            "password": st.session_state.get("auth_password", ""),
            "auth_method": st.session_state.get("auth_method", "Basic"),
            "use_proxy": st.session_state.get("use_proxy", False),
            "timeout": st.session_state.get("timeout", 10),
            "retry_attempts": st.session_state.get("retry_attempts", 3)
        }
        
        # Poll the monitor for vital signs
        new_data = poll_zoll_monitor(config)
        
        if new_data:
            st.session_state.last_update_time = current_time
            st.session_state.connection_status = "Connected"
            st.session_state.last_successful_poll = current_time
        else:
            st.session_state.connection_status = "Disconnected"
    
    elif st.session_state.data_source == "websocket":
        # For WebSocket, check if enough time has passed since last update
        if 'last_update_time' in st.session_state and (current_time - st.session_state.last_update_time).total_seconds() < 1:  # Limit to 1 second
            return
            
        # Fetch data from WebSocket
        data = fetch_websocket_data(st.session_state.get("websocket_url", "ws://localhost:8765"))
        
        # If we got valid data, add it to the streaming data
        if data:
            st.session_state.streaming_data.append(data)
            st.session_state.last_update_time = current_time
            
    elif st.session_state.data_source == "file_import":
        # For file import, we don't update automatically
        pass
    
    # Add the new data to the streaming data
    if new_data:
        new_df = pd.DataFrame([new_data])
        
        # If we already have data, append the new data
        if not st.session_state.streaming_data.empty:
            st.session_state.streaming_data = pd.concat([st.session_state.streaming_data, new_df])
        else:
            st.session_state.streaming_data = new_df
        
        return True
    
    return False

def display_vital_sign_box(title, value, unit, status, normal_range=None, thresholds=None):
    """
    Display a vital sign in a styled box
    
    Args:
        title: Name of the vital sign
        value: Current value
        unit: Unit of measurement
        status: Status of the reading (valid, invalid, etc.)
        normal_range: Tuple with (min, max) normal range
        thresholds: Scenario-specific thresholds
    """
    # Determine the color based on status and thresholds
    if status != "valid":
        color = "#888888"  # Gray for invalid/unmonitored
        background = "#F0F0F0"
    elif thresholds:
        min_val, max_val = thresholds
        if (min_val is not None and value < min_val) or (max_val is not None and value > max_val):
            color = "#FF4B4B"  # Red for out of threshold
            background = "#FFEEEE"
        else:
            color = "#00CC00"  # Green for within threshold
            background = "#EEFFEE"
    elif normal_range:
        min_val, max_val = normal_range
        if (min_val is not None and value < min_val) or (max_val is not None and value > max_val):
            color = "#FFA500"  # Orange for out of normal range
            background = "#FFF8E8"
        else:
            color = "#00CC00"  # Green for normal
            background = "#EEFFEE"
    else:
        color = "#00CC00"  # Default green
        background = "#EEFFEE"
    
    # Create the HTML for the vital sign box
    html = f"""
    <div style="
        background-color: {background};
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    ">
        <h3 style="margin: 0; font-size: 1rem; color: #555;">{title}</h3>
        <div style="
            font-size: 2rem;
            font-weight: bold;
            color: {color};
            margin: 5px 0;
        ">
            {value} <span style="font-size: 1rem;">{unit}</span>
        </div>
    """
    
    # Add threshold or normal range information if available
    if thresholds:
        min_val, max_val = thresholds
        range_text = ""
        if min_val is not None and max_val is not None:
            range_text = f"Target: {min_val}-{max_val}"
        elif min_val is not None:
            range_text = f"Target: ≥{min_val}"
        elif max_val is not None:
            range_text = f"Target: ≤{max_val}"
        
        if range_text:
            html += f'<div style="font-size: 0.8rem; color: #777;">{range_text}</div>'
    
    html += "</div>"
    
    return html

def display_real_time_monitor():
    """Display the real-time vital signs monitor"""
    # Update the streaming data if active
    if st.session_state.streaming_active:
        update_streaming_data()
    
    # Get the streaming data
    df = st.session_state.streaming_data
    
    if df.empty:
        st.warning("No data available. Waiting for data...")
        return
    
    # Get the most recent data point
    latest_data = df.iloc[-1]
    
    # Get the scenario thresholds
    scenario = st.session_state.get("real_time_scenario", "TBI")
    thresholds = CPG_THRESHOLDS.get(scenario, {})
    
    # Create the monitor layout
    st.write(f"### Patient Monitor - {scenario} Scenario")
    st.write(f"Last updated: {latest_data.get('TimeStr', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}")
    
    # Create a grid layout for vital signs
    col1, col2, col3, col4 = st.columns(4)
    
    # Display vital signs in the grid
    with col1:
        # Heart Rate
        hr_value = latest_data.get("Hr")
        hr_status = latest_data.get("Hr_status", "valid")
        hr_thresholds = thresholds.get("Hr")
        
        if hr_value is not None:
            st.markdown(
                display_vital_sign_box(
                    "Heart Rate", 
                    int(hr_value), 
                    "bpm", 
                    hr_status,
                    normal_range=(60, 100),
                    thresholds=hr_thresholds
                ),
                unsafe_allow_html=True
            )
        
        # Temperature
        temp_value = latest_data.get("Temp1")
        temp_status = latest_data.get("Temp1_status", "valid")
        temp_thresholds = thresholds.get("TempF")
        
        if temp_value is not None:
            st.markdown(
                display_vital_sign_box(
                    "Temperature", 
                    round(temp_value, 1), 
                    "°F", 
                    temp_status,
                    normal_range=(97, 99),
                    thresholds=temp_thresholds
                ),
                unsafe_allow_html=True
            )
    
    with col2:
        # SpO2
        spo2_value = latest_data.get("SpO2")
        spo2_status = latest_data.get("SpO2_status", "valid")
        spo2_thresholds = thresholds.get("SpO2")
        
        if spo2_value is not None:
            st.markdown(
                display_vital_sign_box(
                    "SpO2", 
                    int(spo2_value), 
                    "%", 
                    spo2_status,
                    normal_range=(95, 100),
                    thresholds=spo2_thresholds
                ),
                unsafe_allow_html=True
            )
        
        # Respiratory Rate
        rr_value = latest_data.get("RespRate")
        rr_status = latest_data.get("RespRate_status", "valid")
        rr_thresholds = thresholds.get("RespRate")
        
        if rr_value is not None:
            st.markdown(
                display_vital_sign_box(
                    "Resp Rate", 
                    int(rr_value), 
                    "bpm", 
                    rr_status,
                    normal_range=(12, 20),
                    thresholds=rr_thresholds
                ),
                unsafe_allow_html=True
            )
    
    with col3:
        # Blood Pressure (Systolic/Diastolic)
        sys_value = latest_data.get("NIBP_SYS")
        dia_value = latest_data.get("NIBP_DIA")
        bp_status = latest_data.get("NIBP_SYS_status", "valid")
        sys_thresholds = thresholds.get("NIBP_SYS")
        dia_thresholds = thresholds.get("NIBP_DIA")
        
        if sys_value is not None and dia_value is not None:
            st.markdown(
                display_vital_sign_box(
                    "Blood Pressure", 
                    f"{int(sys_value)}/{int(dia_value)}", 
                    "mmHg", 
                    bp_status,
                    normal_range=None,
                    thresholds=None  # Complex to display for combined value
                ),
                unsafe_allow_html=True
            )
        
        # EtCO2
        etco2_value = latest_data.get("EtCO2")
        etco2_status = latest_data.get("EtCO2_status", "valid")
        etco2_thresholds = thresholds.get("EtCO2")
        
        if etco2_value is not None:
            st.markdown(
                display_vital_sign_box(
                    "EtCO2", 
                    int(etco2_value), 
                    "mmHg", 
                    etco2_status,
                    normal_range=(35, 45),
                    thresholds=etco2_thresholds
                ),
                unsafe_allow_html=True
            )
    
    with col4:
        # MAP
        map_value = latest_data.get("NIBP_MAP")
        map_status = latest_data.get("NIBP_MAP_status", "valid")
        map_thresholds = thresholds.get("NIBP_MAP")
        
        if map_value is not None:
            st.markdown(
                display_vital_sign_box(
                    "MAP", 
                    int(map_value), 
                    "mmHg", 
                    map_status,
                    normal_range=(70, 105),
                    thresholds=map_thresholds
                ),
                unsafe_allow_html=True
            )
    
    # Display real-time trends
    st.write("### Real-Time Trends")
    
    # Limit to the last 60 data points for performance
    trend_df = df.tail(60).copy()
    
    # Create time axis for plotting
    if "TimeObj" in trend_df.columns:
        trend_df["ElapsedMin"] = (trend_df["TimeObj"] - trend_df["TimeObj"].iloc[0]).dt.total_seconds() / 60.0
    else:
        # Create a time index if TimeObj is not available
        trend_df["ElapsedMin"] = range(len(trend_df))
    
    # Create tabs for different vital sign groups
    trend_tabs = st.tabs(["Cardiovascular", "Respiratory", "Temperature"])
    
    with trend_tabs[0]:
        # Cardiovascular vital signs
        cv_vitals = ["Hr", "NIBP_SYS", "NIBP_DIA", "NIBP_MAP"]
        cv_vitals = [v for v in cv_vitals if v in trend_df.columns]
        
        if cv_vitals:
            # Convert columns to numeric
            for v in cv_vitals:
                trend_df[v] = pd.to_numeric(trend_df[v], errors="coerce")
            
            # Create the plot
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add heart rate
            if "Hr" in cv_vitals:
                fig.add_trace(
                    plotly_go.Scatter(
                        x=trend_df["ElapsedMin"],
                        y=trend_df["Hr"],
                        name="Heart Rate",
                        line=dict(color="red", width=2)
                    )
                )
            
            # Add blood pressure
            if "NIBP_SYS" in cv_vitals and "NIBP_DIA" in cv_vitals:
                fig.add_trace(
                    plotly_go.Scatter(
                        x=trend_df["ElapsedMin"],
                        y=trend_df["NIBP_SYS"],
                        name="Systolic BP",
                        line=dict(color="blue", width=2)
                    )
                )
                fig.add_trace(
                    plotly_go.Scatter(
                        x=trend_df["ElapsedMin"],
                        y=trend_df["NIBP_DIA"],
                        name="Diastolic BP",
                        line=dict(color="lightblue", width=2)
                    )
                )
            
            # Add MAP
            if "NIBP_MAP" in cv_vitals:
                fig.add_trace(
                    plotly_go.Scatter(
                        x=trend_df["ElapsedMin"],
                        y=trend_df["NIBP_MAP"],
                        name="MAP",
                        line=dict(color="purple", width=2)
                    )
                )
            
            # Update layout
            fig.update_layout(
                title="Cardiovascular Trends",
                xaxis_title="Time (minutes)",
                yaxis_title="Value",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=400,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with trend_tabs[1]:
        # Respiratory vital signs
        resp_vitals = ["SpO2", "RespRate", "EtCO2"]
        resp_vitals = [v for v in resp_vitals if v in trend_df.columns]
        
        if resp_vitals:
            # Convert columns to numeric
            for v in resp_vitals:
                trend_df[v] = pd.to_numeric(trend_df[v], errors="coerce")
            
            # Create the plot
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add SpO2
            if "SpO2" in resp_vitals:
                fig.add_trace(
                    plotly_go.Scatter(
                        x=trend_df["ElapsedMin"],
                        y=trend_df["SpO2"],
                        name="SpO2",
                        line=dict(color="blue", width=2)
                    ),
                    secondary_y=False
                )
            
            # Add respiratory rate
            if "RespRate" in resp_vitals:
                fig.add_trace(
                    plotly_go.Scatter(
                        x=trend_df["ElapsedMin"],
                        y=trend_df["RespRate"],
                        name="Resp Rate",
                        line=dict(color="green", width=2)
                    ),
                    secondary_y=True
                )
            
            # Add EtCO2
            if "EtCO2" in resp_vitals:
                fig.add_trace(
                    plotly_go.Scatter(
                        x=trend_df["ElapsedMin"],
                        y=trend_df["EtCO2"],
                        name="EtCO2",
                        line=dict(color="orange", width=2)
                    ),
                    secondary_y=True
                )
            
            # Update layout
            fig.update_layout(
                title="Respiratory Trends",
                xaxis_title="Time (minutes)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=400,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            # Update y-axis labels
            fig.update_yaxes(title_text="SpO2 (%)", secondary_y=False)
            fig.update_yaxes(title_text="Rate/EtCO2", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
    
    with trend_tabs[2]:
        # Temperature
        temp_vitals = ["Temp1"]
        temp_vitals = [v for v in temp_vitals if v in trend_df.columns]
        
        if temp_vitals:
            # Convert columns to numeric
            for v in temp_vitals:
                trend_df[v] = pd.to_numeric(trend_df[v], errors="coerce")
            
            # Create the plot
            fig = plotly_go.Figure()
            
            # Add temperature
            if "Temp1" in temp_vitals:
                fig.add_trace(
                    plotly_go.Scatter(
                        x=trend_df["ElapsedMin"],
                        y=trend_df["Temp1"],
                        name="Temperature",
                        line=dict(color="red", width=2)
                    )
                )
            
            # Update layout
            fig.update_layout(
                title="Temperature Trend",
                xaxis_title="Time (minutes)",
                yaxis_title="Temperature (°F)",
                height=400,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Real-time CPG compliance
    st.markdown("### Real-Time CPG Compliance")
    
    # Add device and scenario information
    trend_df["DevSerial"] = "SIMULATOR"
    trend_df["Scenario"] = scenario
    trend_df["IsValid"] = True
    trend_df["InSimWindow"] = True
    
    # Compute compliance
    compliance_result = compute_time_out_of_range(trend_df, scenario)
    
    # Display compliance score
    st.write(f"**Overall Compliance Score**: {compliance_result['compliance_score']}%")
    
    # Display color-coded assessment
    assessment_color = {
        "EXCELLENT": "success",
        "GOOD": "info",
        "FAIR": "warning",
        "POOR": "error"
    }
    
    # Check if assessment key exists in compliance_result
    if 'assessment' in compliance_result:
        status = compliance_result['assessment'].split(' - ')[0]
        st.markdown(f"**Assessment**: :{assessment_color[status]}[{compliance_result['assessment']}]")
    else:
        st.markdown("**Assessment**: :error[Unable to determine assessment - insufficient data]")
    
    # Display vital signs details in a table
    vital_rows = []
    for vital, stats in compliance_result['vital_stats'].items():
        vital_rows.append({
            "Vital Sign": vital,
            "Target Range": stats['target'],
            "Actual Range": stats['range'],
            "Mean Value": stats['mean_value'],
            "Compliance": f"{stats['in_range_percent']}%",
            "Status": stats['status']
        })
    
    if vital_rows:
        st.write("**Vital Signs Compliance**")
        st.dataframe(pd.DataFrame(vital_rows))
    
    # Option to save the current data
    if st.button("Save Current Session Data"):
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vital_signs_session_{timestamp}.csv"
        
        # Convert the DataFrame to CSV
        csv = df.to_csv(index=False)
        
        # Provide a download button
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=filename,
            mime="text/csv"
        )

def display_monitor_preview():
    """Display a preview of the monitor layout with sample data"""
    # Create sample data
    sample_data = {
        "Hr": 75,
        "SpO2": 98,
        "NIBP_SYS": 120,
        "NIBP_DIA": 80,
        "NIBP_MAP": 93,
        "Temp1": 98.6,
        "RespRate": 16,
        "EtCO2": 35,
        "Hr_status": "valid",
        "SpO2_status": "valid",
        "NIBP_SYS_status": "valid",
        "NIBP_DIA_status": "valid",
        "NIBP_MAP_status": "valid",
        "Temp1_status": "valid",
        "RespRate_status": "valid",
        "EtCO2_status": "valid"
    }
    
    # Create the monitor layout
    st.write("This is a preview of how the monitor will look when streaming data.")
    
    # Create a grid layout for vital signs
    col1, col2, col3, col4 = st.columns(4)
    
    # Display vital signs in the grid
    with col1:
        st.markdown(
            display_vital_sign_box(
                "Heart Rate", 
                sample_data["Hr"], 
                "bpm", 
                "valid",
                normal_range=(60, 100)
            ),
            unsafe_allow_html=True
        )
        
        st.markdown(
            display_vital_sign_box(
                "Temperature", 
                sample_data["Temp1"], 
                "°F", 
                "valid",
                normal_range=(97, 99)
            ),
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            display_vital_sign_box(
                "SpO2", 
                sample_data["SpO2"], 
                "%", 
                "valid",
                normal_range=(95, 100)
            ),
            unsafe_allow_html=True
        )
        
        st.markdown(
            display_vital_sign_box(
                "Resp Rate", 
                sample_data["RespRate"], 
                "bpm", 
                "valid",
                normal_range=(12, 20)
            ),
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            display_vital_sign_box(
                "Blood Pressure", 
                f"{sample_data['NIBP_SYS']}/{sample_data['NIBP_DIA']}", 
                "mmHg", 
                "valid"
            ),
            unsafe_allow_html=True
        )
        
        st.markdown(
            display_vital_sign_box(
                "EtCO2", 
                sample_data["EtCO2"], 
                "mmHg", 
                "valid",
                normal_range=(35, 45)
            ),
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            display_vital_sign_box(
                "MAP", 
                sample_data["NIBP_MAP"], 
                "mmHg", 
                "valid",
                normal_range=(70, 105)
            ),
            unsafe_allow_html=True
        )
    
    # Display sample trend chart
    st.write("### Sample Trend Chart")
    st.image("https://www.researchgate.net/profile/Omer-Inan/publication/224142028/figure/fig1/AS:302600435404802@1449154410873/Vital-signs-monitor-display-showing-the-ECG-trace-top-the-pulse-oximeter-trace-middle.png", 
             caption="Sample vital signs monitor display")

###############################################################################
# ZOLL MONITOR INTEGRATION
###############################################################################

class ZollMonitorClient:
    """Client for connecting to and retrieving data from Zoll Propaq monitors"""
    
    def __init__(self, ip_address, port=80, connection_method="HTTP Polling", 
                 auth_required=False, username="", password="", auth_method="Basic",
                 use_proxy=False, timeout=10, retry_attempts=3):
        """
        Initialize the Zoll Monitor client
        
        Args:
            ip_address: IP address of the monitor
            port: Port number (default: 80)
            connection_method: Method to connect to the monitor
            auth_required: Whether authentication is required
            username: Authentication username
            password: Authentication password
            auth_method: Authentication method (Basic, Digest, Token)
            use_proxy: Whether to use a proxy server
            timeout: Connection timeout in seconds
            retry_attempts: Number of retry attempts
        """
        self.ip_address = ip_address
        self.port = port
        self.connection_method = connection_method
        self.auth_required = auth_required
        self.username = username
        self.password = password
        self.auth_method = auth_method
        self.use_proxy = use_proxy
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.connected = False
        self.last_update_time = None
        self.session = requests.Session() 
    
    def get_base_url(self):
        """Get the base URL for the monitor"""
        return f"http://{self.ip_address}:{self.port}"
    
    def get_auth(self):
        """Get the authentication object based on the auth method"""
        if not self.auth_required:
            return None
        
        if self.auth_method == "Basic":
            return HTTPBasicAuth(self.username, self.password)
        elif self.auth_method == "Digest":
            return HTTPDigestAuth(self.username, self.password)
        elif self.auth_method == "Token":
            # Token auth is handled in headers
            return None
        
        return None
    
    def get_headers(self):
        """Get the headers for requests"""
        headers = {
            "User-Agent": "VitalsApp/1.5",
            "Accept": "application/json"
        }
        
        # Add token auth if needed
        if self.auth_required and self.auth_method == "Token":
            headers["Authorization"] = f"Bearer {self.password}"
        
        return headers
    
    def test_connection(self):
        """Test the connection to the monitor"""
        try:
            if self.connection_method == "HTTP Polling":
                url = f"{self.get_base_url()}/api/status"
                response = self.session.get(
                    url,
                    auth=self.get_auth(),
                    headers=self.get_headers(),
                    timeout=self.timeout,
                    verify=False  # Skip SSL verification
                )
                
                if response.status_code == 200:
                    self.connected = True
                    return True, "Connected successfully"
                else:
                    return False, f"Failed to connect: HTTP {response.status_code}"
            
            elif self.connection_method == "TCP Direct":
                # Simulate TCP connection
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.timeout)
                result = sock.connect_ex((self.ip_address, self.port))
                sock.close()
                
                if result == 0:
                    self.connected = True
                    return True, "Connected successfully"
                else:
                    return False, f"Failed to connect: TCP error code {result}"
            
            elif self.connection_method == "UDP Stream":
                # Simulate UDP connection
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.settimeout(self.timeout)
                sock.sendto(b"HELLO", (self.ip_address, self.port))
                sock.close()
                self.connected = True
                return True, "UDP stream initialized"
            
            elif self.connection_method == "File Export":
                # Simulate file export check
                self.connected = True
                return True, "File export path accessible"
            
            return False, "Unknown connection method"
        
        except Exception as e:
            return False, f"Connection error: {str(e)}"
    
    def poll_vital_signs(self):
        """
        Poll the monitor for vital signs data
        
        Returns:
            Dictionary with vital signs data or None if failed
        """
        if not self.connected:
            success, message = self.test_connection()
            if not success:
                return None
        
        try:
            if self.connection_method == "HTTP Polling":
                # In a real implementation, this would make an actual HTTP request
                # to the monitor's API endpoint for vital signs data
                
                # Simulate a successful response with sample data
                vital_data = generate_simulated_vitals()
                self.last_update_time = datetime.now()
                return vital_data
            
            elif self.connection_method == "TCP Direct":
                # Simulate TCP data retrieval
                vital_data = generate_simulated_vitals()
                self.last_update_time = datetime.now()
                return vital_data
            
            elif self.connection_method == "UDP Stream":
                # Simulate UDP data retrieval
                vital_data = generate_simulated_vitals()
                self.last_update_time = datetime.now()
                return vital_data
            
            elif self.connection_method == "File Export":
                # Simulate file export data retrieval
                vital_data = generate_simulated_vitals()
                self.last_update_time = datetime.now()
                return vital_data
            
            return None
        
        except Exception as e:
            logging.error(f"Error polling vital signs: {str(e)}")
            return None
    
    def disconnect(self):
        """Disconnect from the monitor"""
        self.connected = False
        self.session.close()
        return True

def poll_zoll_monitor(config):
    """
    Poll a Zoll monitor for vital signs data
    
    Args:
        config: Dictionary with monitor configuration
        
    Returns:
        Dictionary with vital signs data or None if failed
    """
    # Create a client
    client = ZollMonitorClient(
        ip_address=config.get("ip_address", "192.168.1.100"),
        port=config.get("port", 80),
        connection_method=config.get("connection_method", "HTTP Polling"),
        auth_required=config.get("auth_required", False),
        username=config.get("username", ""),
        password=config.get("password", ""),
        auth_method=config.get("auth_method", "Basic"),
        use_proxy=config.get("use_proxy", False),
        timeout=config.get("timeout", 10),
        retry_attempts=config.get("retry_attempts", 3)
    )
    
    # Poll for vital signs
    vital_data = client.poll_vital_signs()
    
    # Disconnect
    client.disconnect()
    
    return vital_data

# ... existing code ...

def test_time_module():
    """Test function to ensure time_module is working correctly"""
    time_module.sleep(0.1)
    return True

if __name__ == "__main__":
    # Launch the main app
    main()
