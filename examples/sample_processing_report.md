# Excel Processing Report
**File**: `financial_data_2024.xlsx`
**Processed**: 2024-01-15 14:30:15
**Overall Status**: ✅ SUCCESS

## 📊 Executive Summary

- **Total Worksheets**: 3
- **Worksheets Passed**: 2 ✅
- **Worksheets Failed**: 1 ❌
- **CSV Files Generated**: 2
- **Total Rows Processed**: 1,550
- **Processing Time**: 2,456.30ms
- **Confidence Threshold**: 75.00%

## 📋 Worksheet Analysis Details

### 1. Worksheet: `Summary` ✅
**Status**: PASSED

**📈 Data Metrics**:
- Rows: 50
- Columns: 6
- Non-empty cells: 275
- Data density: 92.00%
- Processing time: 85.30ms

**🎯 Confidence Analysis**:
- Overall score: 0.880
- Data density score: 0.920
- Header quality score: 0.850
- Consistency score: 0.870
- Threshold: 0.750
- Is confident: Yes

**📝 Analysis Reasons**:
- High data density detected
- Clear column headers found
- Consistent data types across columns

**📑 Header Information**:
- Has headers: Yes
- Header row: 0
- Header quality: 0.850
- Column names: Date, Revenue, Expenses, Profit, Region, Category

**📄 Generated CSV**:
- File: `financial_data_2024_Summary.csv`
- Size: 8.0 KB

---

### 2. Worksheet: `Raw Data` ✅
**Status**: PASSED

**📈 Data Metrics**:
- Rows: 1,500
- Columns: 12
- Non-empty cells: 16,800
- Data density: 93.00%
- Processing time: 342.70ms

**🎯 Confidence Analysis**:
- Overall score: 0.910
- Data density score: 0.930
- Header quality score: 0.890
- Consistency score: 0.910
- Threshold: 0.750
- Is confident: Yes

**📝 Analysis Reasons**:
- Excellent data density
- Professional column headers
- Highly consistent data structure

**📄 Generated CSV**:
- File: `financial_data_2024_Raw_Data.csv`
- Size: 240.0 KB

---

### 3. Worksheet: `Notes` ❌
**Status**: FAILED

**📈 Data Metrics**:
- Rows: 20
- Columns: 2
- Non-empty cells: 15
- Data density: 37.50%
- Processing time: 25.10ms

**🎯 Confidence Analysis**:
- Overall score: 0.450
- Data density score: 0.375
- Header quality score: 0.300
- Consistency score: 0.675
- Threshold: 0.750
- Is confident: No

**📝 Analysis Reasons**:
- Low data density detected
- Poor header quality
- Mostly textual content

**⚠️ Issues Found**:
- Low data density below threshold
- Contains mostly narrative text rather than structured data
- Insufficient data for meaningful CSV conversion

---

## 📄 CSV Files Generated

Total CSV files: **2**

### 1. `financial_data_2024_Summary.csv` ✅
- **Source worksheet**: Summary
- **Rows written**: 50
- **Columns written**: 6
- **File size**: 8.0 KB
- **Encoding**: utf-8
- **Delimiter**: `','`
- **Generation time**: 15.20ms

### 2. `financial_data_2024_Raw_Data.csv` ✅
- **Source worksheet**: Raw Data
- **Rows written**: 1,500
- **Columns written**: 12
- **File size**: 240.0 KB
- **Encoding**: utf-8
- **Delimiter**: `','`
- **Generation time**: 125.80ms

## ⚡ Performance Metrics

- **Total processing time**: 2,456.30ms
- **Average time per worksheet**: 818.77ms
- **Processing speed**: 631 rows/second
- **Memory efficiency**: 1550 rows processed

## 📦 Archive Information

- **Archive status**: ✅ SUCCESS
- **Source path**: `/data/financial_data_2024.xlsx`
- **Archive path**: `/archive/financial_data_2024_20240115_143000.xlsx`
- **Timestamp used**: 20240115_143000
- **Operation time**: 1.870s

## 🔧 Technical Details

- **Source file**: `/data/financial_data_2024.xlsx`
- **File size**: 2.5 MB
- **Processing timestamp**: 2024-01-15T14:30:00
- **Report generated**: 2024-01-15T14:30:15.456789

### Worksheet Summary Table

| Worksheet | Status | Rows | Columns | Confidence | CSV Generated |
|-----------|--------|------|---------|------------|---------------|
| Summary | ✅ Pass | 50 | 6 | 0.880 | Yes |
| Raw Data | ✅ Pass | 1,500 | 12 | 0.910 | Yes |
| Notes | ❌ Fail | 20 | 2 | 0.450 | No |