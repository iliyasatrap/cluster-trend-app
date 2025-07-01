#!/usr/bin/env python3
"""
Test script to demonstrate the ETL status bar functionality.
This script shows what the status messages look like for different scenarios.
"""

import json
import os
from datetime import datetime, timezone
from dateutil import parser

def load_etl_status(filename="data/etl_status.json"):
    """Load ETL status from JSON file and return formatted status message"""
    try:
        with open(filename) as f:
            status = json.load(f)
        
        # Parse the last run timestamp
        last_run_str = status.get("last_run")
        paper_count = status.get("paper_count", 0)
        
        if not last_run_str:
            return "❌ ETL status unknown - missing timestamp"
        
        # Parse timestamp and calculate time difference
        last_run = parser.parse(last_run_str)
        now = datetime.now(timezone.utc)
        time_diff = now - last_run
        
        # Calculate hours and format time ago
        hours_ago = time_diff.total_seconds() / 3600
        
        if hours_ago < 1:
            minutes_ago = int(time_diff.total_seconds() / 60)
            time_ago = f"{minutes_ago} m ago"
        elif hours_ago < 24:
            hours = int(hours_ago)
            minutes = int((hours_ago - hours) * 60)
            time_ago = f"{hours} h {minutes} m ago"
        else:
            days_ago = int(hours_ago / 24)
            remaining_hours = int(hours_ago % 24)
            time_ago = f"{days_ago} d {remaining_hours} h ago"
        
        # Format paper count with spaces for thousands
        formatted_count = f"{paper_count:,}".replace(",", " ")
        
        # Determine status color and icon
        if hours_ago < 24:
            icon = "✅"
            status_text = "Data fresh"
            color = "GREEN"
        elif hours_ago < 48:
            icon = "⚠️"
            status_text = "Data aging"
            color = "YELLOW"
        else:
            icon = "❌"
            status_text = "Data stale"
            color = "RED"
        
        message = f"{icon} {status_text} ‧ updated {time_ago} ‧ {formatted_count} papers"
        
        return f"[{color}] {message}"
        
    except FileNotFoundError:
        return "[RED] ❌ ETL status unknown - status file not found"
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return f"[RED] ❌ ETL status unknown - malformed status file: {e}"

if __name__ == "__main__":
    print("ETL Status Bar Test")
    print("=" * 50)
    
    # Test current status
    print("\n1. Current status (data/etl_status.json):")
    print(load_etl_status("data/etl_status.json"))
    
    # Test old status
    print("\n2. Old status (data/etl_status_old.json):")
    print(load_etl_status("data/etl_status_old.json"))
    
    # Test missing file
    print("\n3. Missing file:")
    print(load_etl_status("data/nonexistent.json"))
    
    # Test malformed JSON
    print("\n4. Creating and testing malformed JSON:")
    with open("data/etl_status_bad.json", "w") as f:
        f.write('{"last_run": "invalid-date", "paper_count":')
    print(load_etl_status("data/etl_status_bad.json"))
    
    # Clean up
    if os.path.exists("data/etl_status_bad.json"):
        os.remove("data/etl_status_bad.json")
    
    print("\n" + "=" * 50)
    print("The Streamlit app is running at: http://localhost:8503")
    print("You should see a status bar at the top showing the current ETL status.")
