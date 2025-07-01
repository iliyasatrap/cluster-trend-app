# ETL Status Bar Feature

## Overview

A status bar has been added to the top of the Streamlit app that displays the current ETL (Extract, Transform, Load) status, showing when the data was last updated and how many papers are in the dataset.

## Features

### Status Indicators
- **✅ Green**: Data is fresh (updated < 24 hours ago)
- **⚠️ Yellow**: Data is aging (updated 24-48 hours ago)  
- **❌ Red**: Data is stale (updated > 48 hours ago) or status unknown

### Message Format
The status bar displays information in this format:
```
✅ Data fresh ‧ updated 2 h 15 m ago ‧ 35 642 papers
```

### Error Handling
- Missing status file: Shows red error message
- Malformed JSON: Shows red error message with details
- Missing timestamp: Shows red error message
- Any other errors: Shows generic red error message

## Technical Implementation

### Dependencies Added
- `python-dateutil`: For robust timestamp parsing

### Status File Location
The ETL process should write status information to:
```
data/etl_status.json
```

### Expected JSON Format
```json
{
  "last_run": "2025-07-01T17:35:42Z",
  "paper_count": 35642
}
```

Where:
- `last_run`: ISO 8601 timestamp in UTC
- `paper_count`: Integer count of papers processed

### Code Structure
- Status bar logic is implemented in the `load_etl_status()` function
- Status bar is displayed at the very top of the app using `st.markdown()` with custom CSS
- Color-coded styling with left border and background tinting
- Graceful error handling for all failure scenarios

## Testing

Run the test script to see different status scenarios:
```bash
python test_status_bar.py
```

This will show examples of:
1. Fresh data (green status)
2. Aging data (yellow status) 
3. Missing file (red error)
4. Malformed JSON (red error)

## Integration with n8n ETL

Your n8n workflow should write the status file after successful completion:

```javascript
// Example n8n node to write status
const statusData = {
  "last_run": new Date().toISOString(),
  "paper_count": $('previous_node').data.length
};

// Write to data/etl_status.json
return [{ json: statusData }];
```

## Visual Design

The status bar uses:
- Subtle background color matching the status (green/yellow/red with 15% opacity)
- Solid left border in the status color
- Rounded corners and padding for a polished look
- Consistent typography and spacing
- Positioned above all other content

## Time Formatting

Time differences are displayed in a human-readable format:
- `< 1 hour`: "15 m ago"
- `< 24 hours`: "2 h 15 m ago"  
- `≥ 24 hours`: "1 d 5 h ago"

Paper counts are formatted with spaces as thousands separators (e.g., "35 642" instead of "35,642").
