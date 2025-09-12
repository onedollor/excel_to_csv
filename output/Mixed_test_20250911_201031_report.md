# Excel Processing Report
**File**: `Mixed_test.xlsx`
**Processed**: 2025-09-11 20:10:31
**Overall Status**: ✅ SUCCESS

## 📊 Executive Summary

- **Total Worksheets**: 3
- **Worksheets Passed**: 1 ✅
- **Worksheets Failed**: 2 ❌
- **CSV Files Generated**: 1
- **Total Rows Processed**: 21
- **Processing Time**: 1755878195848.63ms
- **Confidence Threshold**: 70.00%

## 📋 Worksheet Analysis Details

### 1. Worksheet: `TableLike20Rows` ✅
**Status**: PASSED

**📈 Data Metrics**:
- Rows: 21
- Columns: 3
- Non-empty cells: 63
- Data density: 100.00%
- Processing time: 50.00ms

**🎯 Confidence Analysis**:
- Overall score: 0.855
- Data density score: 1.000
- Header quality score: 1.000
- Consistency score: 0.517
- Threshold: 0.700
- Is confident: Yes

**📝 Analysis Reasons**:
- Good data density: 1.000
- High quality headers detected (row 0)
- Low consistency across 33.3% of columns

**📄 Generated CSV**:
- File: `Mixed_test_TableLike20Rows.csv`
- Size: 345.0 B

---

### 2. Worksheet: `TableLike` ❌
**Status**: FAILED

**📈 Data Metrics**:
- Rows: 4
- Columns: 3
- Non-empty cells: 12
- Data density: 100.00%
- Processing time: 50.00ms

**🎯 Confidence Analysis**:
- Overall score: 0.000
- Data density score: 0.000
- Header quality score: 0.000
- Consistency score: 0.000
- Threshold: 0.700
- Is confident: No

**📝 Analysis Reasons**:
- Too few rows: 4 < 5

---

### 3. Worksheet: `NoHeaders` ❌
**Status**: FAILED

**📈 Data Metrics**:
- Rows: 4
- Columns: 3
- Non-empty cells: 7
- Data density: 58.33%
- Processing time: 50.00ms

**🎯 Confidence Analysis**:
- Overall score: 0.000
- Data density score: 0.000
- Header quality score: 0.000
- Consistency score: 0.000
- Threshold: 0.700
- Is confident: No

**📝 Analysis Reasons**:
- Too few rows: 4 < 5

---

## 📄 CSV Files Generated

Total CSV files: **1**

### 1. `Mixed_test_TableLike20Rows.csv` ✅
- **Source worksheet**: TableLike20Rows
- **Rows written**: 21
- **Columns written**: 3
- **File size**: 345.0 B
- **Encoding**: utf-8
- **Delimiter**: `','`
- **Generation time**: 25.00ms

## ⚡ Performance Metrics

- **Total processing time**: 1755878195848.63ms
- **Average time per worksheet**: 585292731949.54ms
- **Processing speed**: 0 rows/second
- **Memory efficiency**: 21 rows processed

## 📦 Archive Information

- **Archive status**: ✅ SUCCESS
- **Source path**: `input/Mixed_test.xlsx`
- **Archive path**: `input/archive/Mixed_test_20250911_201031.xlsx`
- **Operation time**: 0.007s

## 🔧 Technical Details

- **Source file**: `input/Mixed_test.xlsx`
- **File size**: Unknown
- **Processing timestamp**: 2025-09-11T20:10:31.682155
- **Report generated**: 2025-09-11T20:10:31.682377

### Worksheet Summary Table

| Worksheet | Status | Rows | Columns | Confidence | CSV Generated |
|-----------|--------|------|---------|------------|---------------|
| TableLike20Rows | ✅ Pass | 21 | 3 | 0.855 | Yes |
| TableLike | ❌ Fail | 4 | 3 | 0.000 | No |
| NoHeaders | ❌ Fail | 4 | 3 | 0.000 | No |
