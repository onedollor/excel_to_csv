"""Confidence analysis for Excel worksheet data table detection.

This module provides intelligent analysis to determine whether Excel worksheets
contain meaningful data tables using multiple scoring algorithms:
- Data density analysis
- Header detection and quality assessment
- Data consistency evaluation
- Statistical analysis of data patterns
"""

import re
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import numpy as np

from excel_to_csv.models.data_models import WorksheetData, ConfidenceScore, HeaderInfo
from excel_to_csv.utils.logger import get_processing_logger


class ConfidenceAnalyzer:
    """Analyzes Excel worksheets to determine data table confidence.
    
    The ConfidenceAnalyzer uses multiple algorithms to determine whether
    an Excel worksheet contains a meaningful data table:
    - Data density analysis (40% weight)
    - Header detection and quality (30% weight)  
    - Column data consistency (30% weight)
    
    Example:
        >>> analyzer = ConfidenceAnalyzer(threshold=0.9)
        >>> score = analyzer.analyze_worksheet(worksheet_data)
        >>> if score.is_confident:
        ...     print("Worksheet contains a data table")
    
    Attributes:
        threshold: Minimum confidence score to consider worksheet valid
        weights: Dictionary of component weights for scoring
        min_rows: Minimum rows required for analysis
        min_columns: Minimum columns required for analysis
    """
    
    # Default scoring weights
    DEFAULT_WEIGHTS = {
        'data_density': 0.4,
        'header_quality': 0.3,
        'consistency': 0.3,
    }
    
    # Common header patterns
    HEADER_PATTERNS = [
        r'^(id|identifier|key)$',
        r'^(name|title|label)$',
        r'^(date|time|timestamp)$',
        r'^(value|amount|quantity|count)$',
        r'^(status|state|type|category)$',
        r'^(description|notes|comments)$',
        r'^(code|number|ref|reference)$',
    ]
    
    # Data type indicators
    NUMERIC_INDICATORS = {'$', '%', '#', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
    DATE_INDICATORS = {'/', '-', ':', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                      'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'monday', 'tuesday', 
                      'wednesday', 'thursday', 'friday', 'saturday', 'sunday'}
    
    def __init__(
        self, 
        threshold: float = 0.8,
        weights: Optional[Dict[str, float]] = None,
        min_rows: int = 5,
        min_columns: int = 2,
        max_empty_percentage: float = 0.3
    ):
        """Initialize confidence analyzer.
        
        Args:
            threshold: Minimum confidence threshold (0.0 to 1.0)
            weights: Custom weights for scoring components
            min_rows: Minimum rows required for analysis
            min_columns: Minimum columns required for analysis
            max_empty_percentage: Maximum allowed empty cell percentage
        """
        self.threshold = threshold
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.min_rows = min_rows
        self.min_columns = min_columns
        self.max_empty_percentage = max_empty_percentage
        self.logger = get_processing_logger(__name__)
        
        self.logger.info(
            msg=f"ConfidenceAnalyzer initialized with threshold: {str(self.threshold)}"
        )

        # Validate weights sum to 1.0
        if abs(sum(self.weights.values()) - 1.0) > 0.001:
            raise ValueError("Weights must sum to 1.0")
        
        # Compile header patterns
        self._header_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.HEADER_PATTERNS]
    
    def analyze_worksheet(self, worksheet_data: WorksheetData) -> ConfidenceScore:
        """Analyze worksheet data and return confidence score.
        
        Args:
            worksheet_data: WorksheetData object to analyze
            
        Returns:
            ConfidenceScore object with analysis results
        """
        reasons = []
        
        # Basic validation
        if worksheet_data.is_empty:
            return ConfidenceScore(0.0, 0.0, 0.0, 0.0, ["Worksheet is empty"], self.threshold)
        
        if worksheet_data.row_count < self.min_rows:
            return ConfidenceScore(
                0.0, 0.0, 0.0, 0.0, 
                [f"Too few rows: {worksheet_data.row_count} < {self.min_rows}"], 
                self.threshold
            )
        
        if worksheet_data.column_count < self.min_columns:
            return ConfidenceScore(
                0.0, 0.0, 0.0, 0.0,
                [f"Too few columns: {worksheet_data.column_count} < {self.min_columns}"], 
                self.threshold
            )
        
        # Calculate component scores
        data_density_score = self._calculate_data_density_score(worksheet_data, reasons)
        header_quality_score = self._calculate_header_quality_score(worksheet_data, reasons)
        consistency_score = self._calculate_consistency_score(worksheet_data, reasons)
        
        # Calculate overall weighted score
        overall_score = (
            self.weights['data_density'] * data_density_score +
            self.weights['header_quality'] * header_quality_score +
            self.weights['consistency'] * consistency_score
        )
        
        # Create confidence score object
        confidence_score = ConfidenceScore(
            overall_score=overall_score,
            data_density=data_density_score,
            header_quality=header_quality_score,
            consistency_score=consistency_score,
            reasons=reasons,
            threshold=self.threshold
        )
        
        # Log analysis result
        self.logger.log_worksheet_analysis(
            worksheet_data.worksheet_name,
            overall_score,
            overall_score >= self.threshold,
            reasons
        )
        
        return confidence_score
    
    def _calculate_data_density_score(
        self, 
        worksheet_data: WorksheetData, 
        reasons: List[str]
    ) -> float:
        """Calculate data density component score.
        
        Args:
            worksheet_data: Worksheet to analyze
            reasons: List to append reasoning to
            
        Returns:
            Data density score (0.0 to 1.0)
        """
        df = worksheet_data.data
        
        # Calculate basic density
        density = worksheet_data.data_density
        
        # Penalty for too much empty space
        if density < (1.0 - self.max_empty_percentage):
            reasons.append(f"Low data density: {density:.3f}")
            return density * 0.5  # Severe penalty for sparse data
        
        # Bonus for good density
        if density > 0.8:
            reasons.append(f"Good data density: {density:.3f}")
            return min(1.0, density * 1.2)
        
        # Check for empty rows/columns that suggest non-tabular data
        empty_rows = df.isnull().all(axis=1).sum()
        empty_columns = df.isnull().all(axis=0).sum()
        
        row_empty_ratio = empty_rows / len(df) if len(df) > 0 else 0
        col_empty_ratio = empty_columns / len(df.columns) if len(df.columns) > 0 else 0
        
        # Penalty for many empty rows/columns
        if row_empty_ratio > 0.2:
            reasons.append(f"Too many empty rows: {row_empty_ratio:.3f}")
            density *= (1.0 - row_empty_ratio * 0.5)
        
        if col_empty_ratio > 0.2:
            reasons.append(f"Too many empty columns: {col_empty_ratio:.3f}")
            density *= (1.0 - col_empty_ratio * 0.5)
        
        # Check for data clustering (good sign for tables)
        clustering_bonus = self._analyze_data_clustering(df)
        if clustering_bonus > 0:
            reasons.append("Data shows good clustering pattern")
            density = min(1.0, density + clustering_bonus)
        
        return max(0.0, min(1.0, density))
    
    def _analyze_data_clustering(self, df: pd.DataFrame) -> float:
        """Analyze data clustering patterns.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Clustering bonus (0.0 to 0.2)
        """
        try:
            # Look for rectangular blocks of data (typical of tables)
            non_null_mask = df.notnull()
            
            # Find the largest rectangular region of non-null data
            max_rect_area = 0
            rows, cols = df.shape
            
            for i in range(rows):
                for j in range(cols):
                    if non_null_mask.iloc[i, j]:
                        # Find largest rectangle starting at (i, j)
                        area = self._largest_rectangle_from_point(non_null_mask, i, j)
                        max_rect_area = max(max_rect_area, area)
            
            # Calculate bonus based on how much of the data forms a rectangle
            total_non_null = non_null_mask.sum().sum()
            if total_non_null > 0:
                rectangularity = max_rect_area / total_non_null
                return min(0.2, rectangularity * 0.3)
            
        except Exception:
            # If clustering analysis fails, don't penalize
            pass
        
        return 0.0
    
    def _largest_rectangle_from_point(
        self, 
        mask: pd.DataFrame, 
        start_row: int, 
        start_col: int
    ) -> int:
        """Find largest rectangle of True values starting from a point.
        
        Args:
            mask: Boolean DataFrame
            start_row: Starting row index
            start_col: Starting column index
            
        Returns:
            Area of largest rectangle
        """
        rows, cols = mask.shape
        max_area = 0
        
        # Try different rectangle heights
        for end_row in range(start_row, rows):
            if not mask.iloc[end_row, start_col]:
                break
            
            # Find width for this height
            width = 0
            for end_col in range(start_col, cols):
                # Check if entire column slice is True
                if mask.iloc[start_row:end_row+1, end_col].all():
                    width += 1
                else:
                    break
            
            area = (end_row - start_row + 1) * width
            max_area = max(max_area, area)
        
        return max_area
    
    def _calculate_header_quality_score(
        self, 
        worksheet_data: WorksheetData, 
        reasons: List[str]
    ) -> float:
        """Calculate header quality component score.
        
        Args:
            worksheet_data: Worksheet to analyze
            reasons: List to append reasoning to
            
        Returns:
            Header quality score (0.0 to 1.0)
        """
        df = worksheet_data.data
        
        # Detect potential header row
        header_info = self._detect_headers(df)
        
        if not header_info.has_headers:
            reasons.append("No clear header row detected")
            return 0.3  # Tables can exist without clear headers
        
        score = header_info.header_quality
        
        # Bonus for good header patterns
        if score > 0.8:
            reasons.append(f"High quality headers detected (row {header_info.header_row})")
        elif score > 0.6:
            reasons.append(f"Moderate quality headers detected (row {header_info.header_row})")
        else:
            reasons.append(f"Low quality headers detected (row {header_info.header_row})")
        
        return score
    
    def _detect_headers(self, df: pd.DataFrame) -> HeaderInfo:
        """Detect and analyze potential header rows.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            HeaderInfo object with header analysis
        """
        best_score = 0.0
        best_row = None
        best_headers = []
        
        # Check first few rows for potential headers
        max_rows_to_check = min(5, len(df))
        
        for row_idx in range(max_rows_to_check):
            row_data = df.iloc[row_idx]
            score, headers = self._score_potential_headers(row_data, df, row_idx)
            
            if score > best_score:
                best_score = score
                best_row = row_idx
                best_headers = headers
        
        has_headers = best_score > 0.3  # Threshold for header detection
        
        return HeaderInfo(
            has_headers=has_headers,
            header_row=best_row if has_headers else None,
            header_quality=best_score,
            column_names=best_headers if has_headers else []
        )
    
    def _score_potential_headers(
        self, 
        row_data: pd.Series, 
        df: pd.DataFrame, 
        row_idx: int
    ) -> Tuple[float, List[str]]:
        """Score a row as potential headers.
        
        Args:
            row_data: Row to score
            df: Full DataFrame for context
            row_idx: Index of the row
            
        Returns:
            Tuple of (score, header_names)
        """
        headers = []
        scores = []
        
        for col_idx, value in enumerate(row_data):
            header_score, header_name = self._score_header_cell(value, df, row_idx, col_idx)
            headers.append(header_name)
            scores.append(header_score)
        
        # Overall score is average of individual header scores
        # with bonus for consistency
        if not scores:
            return 0.0, []
        
        avg_score = np.mean(scores)
        
        # Bonus for having most cells as valid headers
        valid_headers = sum(1 for s in scores if s > 0.3)
        completeness_bonus = (valid_headers / len(scores)) * 0.2
        
        # Bonus for text-heavy row (headers are usually text)
        text_ratio = sum(1 for h in headers if isinstance(h, str) and len(str(h).strip()) > 0) / len(headers)
        text_bonus = text_ratio * 0.1
        
        total_score = min(1.0, avg_score + completeness_bonus + text_bonus)
        
        return total_score, headers
    
    def _score_header_cell(
        self, 
        value: any, 
        df: pd.DataFrame, 
        row_idx: int, 
        col_idx: int
    ) -> Tuple[float, str]:
        """Score individual cell as potential header.
        
        Args:
            value: Cell value to score
            df: Full DataFrame for context
            row_idx: Row index of cell
            col_idx: Column index of cell
            
        Returns:
            Tuple of (score, cleaned_header_name)
        """
        if pd.isna(value):
            return 0.0, ""
        
        header_str = str(value).strip()
        if not header_str:
            return 0.0, ""
        
        score = 0.0
        
        # Length check (headers are usually short but not too short)
        if 2 <= len(header_str) <= 50:
            score += 0.3
        elif len(header_str) > 50:
            score -= 0.2  # Penalty for very long headers
        
        # Text content check
        if header_str.replace(' ', '').replace('_', '').replace('-', '').isalpha():
            score += 0.4  # Bonus for text-only headers
        
        # Pattern matching for common header types
        for pattern in self._header_patterns:
            if pattern.match(header_str):
                score += 0.2
                break
        
        # Check if it looks like a descriptive label
        if any(word in header_str.lower() for word in ['id', 'name', 'date', 'value', 'code', 'type']):
            score += 0.1
        
        # Penalty for numbers (usually not good headers)
        if header_str.replace('.', '').replace('-', '').isdigit():
            score -= 0.3
        
        # Check consistency with data below
        if row_idx < len(df) - 1:
            column_data = df.iloc[row_idx+1:, col_idx]
            consistency_bonus = self._check_header_data_consistency(header_str, column_data)
            score += consistency_bonus
        
        return max(0.0, min(1.0, score)), header_str
    
    def _check_header_data_consistency(self, header: str, column_data: pd.Series) -> float:
        """Check if header is consistent with column data.
        
        Args:
            header: Header text
            column_data: Data in the column below header
            
        Returns:
            Consistency bonus (0.0 to 0.2)
        """
        header_lower = header.lower()
        
        # Sample some non-null values
        sample_data = column_data.dropna().head(10)
        if len(sample_data) == 0:
            return 0.0
        
        # Check for date headers with date data
        if any(word in header_lower for word in ['date', 'time', 'timestamp']):
            try:
                pd.to_datetime(sample_data.iloc[:3], errors='raise')
                return 0.2  # Strong bonus for date header with date data
            except:
                pass
        
        # Check for numeric headers with numeric data
        if any(word in header_lower for word in ['amount', 'value', 'count', 'number', 'price', 'cost']):
            try:
                pd.to_numeric(sample_data.iloc[:3], errors='raise')
                return 0.1  # Bonus for numeric header with numeric data
            except:
                pass
        
        # Check for ID headers with ID-like data
        if any(word in header_lower for word in ['id', 'identifier', 'key', 'code']):
            # ID data is often short and alphanumeric
            avg_length = np.mean([len(str(val)) for val in sample_data[:3]])
            if 2 <= avg_length <= 20:
                return 0.1
        
        return 0.0
    
    def _calculate_consistency_score(
        self, 
        worksheet_data: WorksheetData, 
        reasons: List[str]
    ) -> float:
        """Calculate data consistency component score.
        
        Args:
            worksheet_data: Worksheet to analyze  
            reasons: List to append reasoning to
            
        Returns:
            Consistency score (0.0 to 1.0)
        """
        df = worksheet_data.data
        
        column_scores = []
        
        for col_idx, col in enumerate(df.columns):
            column_data = df[col].dropna()
            
            if len(column_data) == 0:
                continue  # Skip empty columns
            
            # Score individual column consistency
            col_score = self._score_column_consistency(column_data)
            column_scores.append(col_score)
        
        if not column_scores:
            reasons.append("No data columns to analyze")
            return 0.0
        
        # Overall consistency is average of column scores
        avg_consistency = np.mean(column_scores)
        
        # Bonus for having consistent data types across most columns
        good_columns = sum(1 for score in column_scores if score > 0.7)
        consistency_ratio = good_columns / len(column_scores)
        
        if consistency_ratio > 0.8:
            reasons.append(f"High consistency across {consistency_ratio:.1%} of columns")
            avg_consistency = min(1.0, avg_consistency + 0.1)
        elif consistency_ratio > 0.6:
            reasons.append(f"Moderate consistency across {consistency_ratio:.1%} of columns")
        else:
            reasons.append(f"Low consistency across {consistency_ratio:.1%} of columns")
            avg_consistency *= 0.8
        
        return max(0.0, min(1.0, avg_consistency))
    
    def _score_column_consistency(self, column_data: pd.Series) -> float:
        """Score consistency within a single column.
        
        Args:
            column_data: Non-null data from a single column
            
        Returns:
            Column consistency score (0.0 to 1.0)
        """
        if len(column_data) <= 1:
            return 0.5  # Not enough data to judge
        
        # Convert to strings for analysis
        str_data = column_data.astype(str)
        
        # Check for numeric consistency
        numeric_score = self._check_numeric_consistency(str_data)
        if numeric_score > 0.8:
            return numeric_score
        
        # Check for date consistency
        date_score = self._check_date_consistency(str_data)
        if date_score > 0.8:
            return date_score
        
        # Check for categorical consistency
        categorical_score = self._check_categorical_consistency(str_data)
        
        # Return the best score
        return max(numeric_score, date_score, categorical_score)
    
    def _check_numeric_consistency(self, str_data: pd.Series) -> float:
        """Check if column contains consistent numeric data.
        
        Args:
            str_data: String representation of column data
            
        Returns:
            Numeric consistency score (0.0 to 1.0)
        """
        try:
            # Try to convert to numeric
            numeric_data = pd.to_numeric(str_data, errors='coerce')
            valid_numeric = numeric_data.dropna()
            
            if len(valid_numeric) == 0:
                return 0.0
            
            # Score based on percentage of valid numeric values
            numeric_ratio = len(valid_numeric) / len(str_data)
            
            if numeric_ratio > 0.9:
                return 0.95  # Very consistent
            elif numeric_ratio > 0.8:
                return 0.8
            elif numeric_ratio > 0.6:
                return 0.6
            else:
                return numeric_ratio * 0.5
        
        except:
            return 0.0
    
    def _check_date_consistency(self, str_data: pd.Series) -> float:
        """Check if column contains consistent date data.
        
        Args:
            str_data: String representation of column data
            
        Returns:
            Date consistency score (0.0 to 1.0)
        """
        try:
            # Try to convert to datetime
            date_data = pd.to_datetime(str_data, errors='coerce')
            valid_dates = date_data.dropna()
            
            if len(valid_dates) == 0:
                return 0.0
            
            # Score based on percentage of valid dates
            date_ratio = len(valid_dates) / len(str_data)
            
            if date_ratio > 0.9:
                return 0.9  # Very consistent dates
            elif date_ratio > 0.8:
                return 0.75
            elif date_ratio > 0.6:
                return 0.6
            else:
                return date_ratio * 0.4
        
        except:
            return 0.0
    
    def _check_categorical_consistency(self, str_data: pd.Series) -> float:
        """Check if column contains consistent categorical data.
        
        Args:
            str_data: String representation of column data
            
        Returns:
            Categorical consistency score (0.0 to 1.0)
        """
        # Check value length consistency
        lengths = str_data.str.len()
        length_std = lengths.std()
        length_mean = lengths.mean()
        
        # Consistent lengths suggest structured data
        if length_mean > 0:
            length_cv = length_std / length_mean  # Coefficient of variation
            length_consistency = max(0, 1 - length_cv)
        else:
            length_consistency = 0
        
        # Check for repeated patterns
        unique_values = len(str_data.unique())
        total_values = len(str_data)
        
        # Good categorical data has reasonable number of unique values
        if total_values > 0:
            uniqueness_ratio = unique_values / total_values
            
            # Sweet spot: not too unique (like IDs) but not too repetitive
            if 0.1 <= uniqueness_ratio <= 0.8:
                pattern_score = 0.8
            elif uniqueness_ratio < 0.1:
                pattern_score = 0.6  # Very repetitive, might be categories
            else:
                pattern_score = 0.4  # Too unique, might be free text
        else:
            pattern_score = 0
        
        # Average the consistency measures
        return (length_consistency + pattern_score) / 2
    
    def get_analysis_summary(self, confidence_score: ConfidenceScore) -> str:
        """Get human-readable analysis summary.
        
        Args:
            confidence_score: ConfidenceScore to summarize
            
        Returns:
            Summary string
        """
        summary = f"Overall Confidence: {confidence_score.overall_score:.3f}\n"
        summary += f"  Data Density: {confidence_score.data_density:.3f}\n"
        summary += f"  Header Quality: {confidence_score.header_quality:.3f}\n"
        summary += f"  Data Consistency: {confidence_score.consistency_score:.3f}\n"
        
        if confidence_score.reasons:
            summary += "\nReasons:\n"
            for reason in confidence_score.reasons:
                summary += f"  - {reason}\n"
        
        summary += f"\nDecision: {'ACCEPT' if confidence_score.is_confident else 'REJECT'}"
        
        return summary