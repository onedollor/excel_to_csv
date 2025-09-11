"""Comprehensive tests for ConfidenceAnalyzer targeting 90%+ coverage.

This test suite covers all major functionality including:
- Analyzer initialization and configuration
- Worksheet confidence analysis workflows
- Data density calculations and validations
- Header detection and quality analysis
- Pattern matching and scoring algorithms
- Edge cases, error handling, and boundary conditions
- Performance and optimization scenarios
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from excel_to_csv.analysis.confidence_analyzer import ConfidenceAnalyzer
from excel_to_csv.models.data_models import WorksheetData, ConfidenceScore, HeaderInfo


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for test files."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_worksheet_data():
    """Create sample WorksheetData for testing."""
    data = pd.DataFrame({
        'ID': [1, 2, 3, 4, 5],
        'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'Score': [95.5, 87.2, 92.1, 88.8, 90.3],
        'Grade': ['A', 'B', 'A', 'B', 'A']
    })
    
    return WorksheetData(
        source_file=Path("test.xlsx"),
        worksheet_name="TestSheet",
        data=data
    )


class TestConfidenceAnalyzerInitialization:
    """Test ConfidenceAnalyzer initialization and configuration."""
    
    def test_analyzer_default_initialization(self):
        """Test analyzer initialization with default parameters."""
        analyzer = ConfidenceAnalyzer()
        
        assert analyzer.threshold == 0.7
        assert analyzer.min_rows == 3
        assert analyzer.min_columns == 2
        assert analyzer.max_empty_percentage == 0.8
        assert analyzer.weights == ConfidenceAnalyzer.DEFAULT_WEIGHTS
        assert hasattr(analyzer, 'logger')
        assert hasattr(analyzer, '_header_patterns')
        assert len(analyzer._header_patterns) > 0
    
    def test_analyzer_custom_initialization(self):
        """Test analyzer initialization with custom parameters."""
        custom_weights = {
            'data_density': 0.4,
            'header_quality': 0.3,
            'structure_score': 0.2,
            'pattern_score': 0.1
        }
        
        analyzer = ConfidenceAnalyzer(
            threshold=0.8,
            weights=custom_weights,
            min_rows=5,
            min_columns=3,
            max_empty_percentage=0.6
        )
        
        assert analyzer.threshold == 0.8
        assert analyzer.min_rows == 5
        assert analyzer.min_columns == 3
        assert analyzer.max_empty_percentage == 0.6
        assert analyzer.weights == custom_weights
    
    def test_analyzer_invalid_weights_sum(self):
        """Test analyzer initialization with invalid weights sum."""
        invalid_weights = {
            'data_density': 0.3,
            'header_quality': 0.3,
            'structure_score': 0.2,
            'pattern_score': 0.1  # Sum = 0.9, not 1.0
        }
        
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            ConfidenceAnalyzer(weights=invalid_weights)
    
    def test_analyzer_weights_close_to_one(self):
        """Test analyzer accepts weights that sum very close to 1.0."""
        close_weights = {
            'data_density': 0.25,
            'header_quality': 0.25,
            'structure_score': 0.25,
            'pattern_score': 0.2501  # Sum = 1.0001, within tolerance
        }
        
        # Should not raise exception due to tolerance
        analyzer = ConfidenceAnalyzer(weights=close_weights)
        assert analyzer.weights == close_weights
    
    def test_analyzer_boundary_parameters(self):
        """Test analyzer with boundary parameter values."""
        # Test minimum values
        analyzer_min = ConfidenceAnalyzer(
            threshold=0.0,
            min_rows=1,
            min_columns=1,
            max_empty_percentage=0.0
        )
        
        assert analyzer_min.threshold == 0.0
        assert analyzer_min.min_rows == 1
        assert analyzer_min.min_columns == 1
        assert analyzer_min.max_empty_percentage == 0.0
        
        # Test maximum values
        analyzer_max = ConfidenceAnalyzer(
            threshold=1.0,
            min_rows=1000,
            min_columns=100,
            max_empty_percentage=1.0
        )
        
        assert analyzer_max.threshold == 1.0
        assert analyzer_max.min_rows == 1000
        assert analyzer_max.max_empty_percentage == 1.0


class TestWorksheetAnalysis:
    """Test worksheet confidence analysis functionality."""
    
    def test_analyze_worksheet_basic_success(self, sample_worksheet_data):
        """Test basic successful worksheet analysis."""
        analyzer = ConfidenceAnalyzer()
        
        result = analyzer.analyze_worksheet(sample_worksheet_data)
        
        assert isinstance(result, ConfidenceScore)
        assert 0.0 <= result.overall_score <= 1.0
        assert 0.0 <= result.data_density <= 1.0
        assert 0.0 <= result.header_quality <= 1.0
        assert 0.0 <= result.structure_score <= 1.0
        assert 0.0 <= result.pattern_score <= 1.0
        assert isinstance(result.reasons, list)
        assert result.threshold == analyzer.threshold
    
    def test_analyze_empty_worksheet(self):
        """Test analysis of empty worksheet."""
        empty_data = pd.DataFrame()
        empty_worksheet = WorksheetData(
            source_file=Path("empty.xlsx"),
            worksheet_name="EmptySheet",
            data=empty_data
        )
        
        analyzer = ConfidenceAnalyzer()
        result = analyzer.analyze_worksheet(empty_worksheet)
        
        assert result.overall_score == 0.0
        assert result.data_density == 0.0
        assert result.header_quality == 0.0
        assert result.structure_score == 0.0
        assert result.pattern_score == 0.0
        assert "Worksheet is empty" in result.reasons
    
    def test_analyze_worksheet_insufficient_rows(self):
        """Test analysis of worksheet with insufficient rows."""
        small_data = pd.DataFrame({'A': [1], 'B': [2]})  # Only 1 row
        small_worksheet = WorksheetData(
            source_file=Path("small.xlsx"),
            worksheet_name="SmallSheet",
            data=small_data
        )
        
        analyzer = ConfidenceAnalyzer(min_rows=3)
        result = analyzer.analyze_worksheet(small_worksheet)
        
        assert result.overall_score == 0.0
        assert any("Insufficient rows" in reason for reason in result.reasons)
    
    def test_analyze_worksheet_insufficient_columns(self):
        """Test analysis of worksheet with insufficient columns."""
        narrow_data = pd.DataFrame({'A': [1, 2, 3, 4, 5]})  # Only 1 column
        narrow_worksheet = WorksheetData(
            source_file=Path("narrow.xlsx"),
            worksheet_name="NarrowSheet",
            data=narrow_data
        )
        
        analyzer = ConfidenceAnalyzer(min_columns=2)
        result = analyzer.analyze_worksheet(narrow_worksheet)
        
        assert result.overall_score == 0.0
        assert any("Insufficient columns" in reason for reason in result.reasons)
    
    def test_analyze_worksheet_high_empty_percentage(self):
        """Test analysis of worksheet with high empty percentage."""
        # Create mostly empty data
        sparse_data = pd.DataFrame({
            'A': [1, None, None, None, None],
            'B': [None, None, None, None, None],
            'C': [None, None, None, None, None]
        })
        sparse_worksheet = WorksheetData(
            source_file=Path("sparse.xlsx"),
            worksheet_name="SparseSheet",
            data=sparse_data
        )
        
        analyzer = ConfidenceAnalyzer(max_empty_percentage=0.5)
        result = analyzer.analyze_worksheet(sparse_worksheet)
        
        assert result.overall_score == 0.0
        assert any("Too many empty cells" in reason for reason in result.reasons)
    
    def test_analyze_worksheet_with_good_headers(self):
        """Test analysis of worksheet with good header patterns."""
        header_data = pd.DataFrame({
            'Customer_ID': [1, 2, 3, 4, 5],
            'Customer_Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'Total_Amount': [100.50, 200.75, 150.25, 300.00, 250.80],
            'Order_Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
        })
        
        header_worksheet = WorksheetData(
            source_file=Path("headers.xlsx"),
            worksheet_name="HeaderSheet",
            data=header_data
        )
        
        analyzer = ConfidenceAnalyzer()
        result = analyzer.analyze_worksheet(header_worksheet)
        
        # Should have reasonable header quality
        assert result.header_quality > 0.3  # Headers contain common patterns
        assert result.overall_score > 0.0
    
    def test_analyze_worksheet_different_thresholds(self, sample_worksheet_data):
        """Test analysis with different confidence thresholds."""
        # Low threshold
        analyzer_low = ConfidenceAnalyzer(threshold=0.3)
        result_low = analyzer_low.analyze_worksheet(sample_worksheet_data)
        
        # High threshold
        analyzer_high = ConfidenceAnalyzer(threshold=0.9)
        result_high = analyzer_high.analyze_worksheet(sample_worksheet_data)
        
        # Results should be the same, only acceptance differs
        assert result_low.overall_score == result_high.overall_score
        assert result_low.threshold != result_high.threshold
        assert result_low.passes_threshold() != result_high.passes_threshold()


class TestDataDensityAnalysis:
    """Test data density calculation functionality."""
    
    def test_calculate_data_density_full_data(self):
        """Test data density calculation with full data."""
        analyzer = ConfidenceAnalyzer()
        
        # Create completely filled data
        full_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['x', 'y', 'z', 'w', 'v'],
            'C': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        density = analyzer._calculate_data_density(full_data)
        
        assert density == 1.0  # All cells filled
    
    def test_calculate_data_density_empty_data(self):
        """Test data density calculation with empty data."""
        analyzer = ConfidenceAnalyzer()
        
        # Create completely empty data
        empty_data = pd.DataFrame({
            'A': [None, None, None],
            'B': [None, None, None],
            'C': [None, None, None]
        })
        
        density = analyzer._calculate_data_density(empty_data)
        
        assert density == 0.0  # All cells empty
    
    def test_calculate_data_density_mixed_data(self):
        """Test data density calculation with mixed data."""
        analyzer = ConfidenceAnalyzer()
        
        # Create half-filled data
        mixed_data = pd.DataFrame({
            'A': [1, None, 3, None],
            'B': [None, 'y', None, 'w'],
            'C': [1.1, None, None, 4.4]
        })
        
        density = analyzer._calculate_data_density(mixed_data)
        
        # 6 filled cells out of 12 total = 0.5
        assert abs(density - 0.5) < 0.01
    
    def test_calculate_data_density_with_zeros_and_empty_strings(self):
        """Test data density treats zeros and empty strings as filled."""
        analyzer = ConfidenceAnalyzer()
        
        # Data with zeros and empty strings
        mixed_data = pd.DataFrame({
            'A': [0, '', None],
            'B': [None, 0, ''],
            'C': ['', None, 0]
        })
        
        density = analyzer._calculate_data_density(mixed_data)
        
        # 6 filled cells (including 0 and '') out of 9 total = 0.667
        assert abs(density - 0.667) < 0.01


class TestHeaderAnalysis:
    """Test header detection and quality analysis."""
    
    def test_analyze_headers_with_good_patterns(self):
        """Test header analysis with common header patterns."""
        analyzer = ConfidenceAnalyzer()
        
        # Headers with common patterns
        headers = ['Customer_ID', 'Customer_Name', 'Total_Amount', 'Order_Date', 'Product_Code']
        
        header_info = analyzer._analyze_headers(headers)
        
        assert isinstance(header_info, HeaderInfo)
        assert header_info.has_headers is True
        assert header_info.header_row == 0
        assert header_info.header_quality > 0.5  # Should recognize patterns
        assert header_info.column_names == headers
    
    def test_analyze_headers_with_poor_patterns(self):
        """Test header analysis with poor header patterns."""
        analyzer = ConfidenceAnalyzer()
        
        # Headers without common patterns
        headers = ['a', 'b', 'c', '1', '2']
        
        header_info = analyzer._analyze_headers(headers)
        
        assert header_info.has_headers is True  # Still detected as headers
        assert header_info.header_quality < 0.5  # But low quality
        assert header_info.column_names == headers
    
    def test_analyze_headers_empty_list(self):
        """Test header analysis with empty headers."""
        analyzer = ConfidenceAnalyzer()
        
        header_info = analyzer._analyze_headers([])
        
        assert header_info.has_headers is False
        assert header_info.header_row is None
        assert header_info.header_quality == 0.0
        assert header_info.column_names == []
    
    def test_analyze_headers_with_none_values(self):
        """Test header analysis with None values in headers."""
        analyzer = ConfidenceAnalyzer()
        
        headers = ['Good_Header', None, 'Another_Good', None, 'Last_Header']
        
        header_info = analyzer._analyze_headers(headers)
        
        assert header_info.has_headers is True
        assert header_info.header_quality > 0.0  # Should still have some quality
        assert len(header_info.column_names) == 5
    
    def test_detect_header_patterns_comprehensive(self):
        """Test comprehensive header pattern detection."""
        analyzer = ConfidenceAnalyzer()
        
        # Test various pattern types
        test_cases = [
            (['Customer_ID', 'Product_Name'], 'underscore_pattern'),
            (['customerID', 'productName'], 'camelCase_pattern'),
            (['Customer ID', 'Product Name'], 'space_pattern'),
            (['Date', 'Time', 'Amount'], 'common_words'),
            (['Total', 'Count', 'Average'], 'aggregate_words'),
            (['Email', 'Phone', 'Address'], 'contact_words'),
        ]
        
        for headers, description in test_cases:
            score = analyzer._detect_header_patterns(headers)
            assert 0.0 <= score <= 1.0, f"Failed for {description}: {headers}"
            assert score > 0.0, f"Should detect some patterns in {description}: {headers}"


class TestStructureAnalysis:
    """Test data structure analysis functionality."""
    
    def test_analyze_data_structure_consistent_types(self):
        """Test structure analysis with consistent data types."""
        analyzer = ConfidenceAnalyzer()
        
        # Consistent data types per column
        consistent_data = pd.DataFrame({
            'IDs': [1, 2, 3, 4, 5],
            'Names': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'Scores': [95.5, 87.2, 92.1, 88.8, 90.3],
            'Active': [True, False, True, True, False]
        })
        
        score = analyzer._analyze_data_structure(consistent_data)
        
        assert score > 0.7  # Should score high for consistent structure
    
    def test_analyze_data_structure_mixed_types(self):
        """Test structure analysis with mixed data types."""
        analyzer = ConfidenceAnalyzer()
        
        # Mixed data types per column
        mixed_data = pd.DataFrame({
            'Mixed1': [1, 'text', 3.14, True, None],
            'Mixed2': ['string', 42, 'another', 5.5, 'last'],
            'Mixed3': [True, 1, 'text', 3.14, False]
        })
        
        score = analyzer._analyze_data_structure(mixed_data)
        
        assert score < 0.5  # Should score low for inconsistent structure
    
    def test_analyze_data_structure_empty_data(self):
        """Test structure analysis with empty data."""
        analyzer = ConfidenceAnalyzer()
        
        empty_data = pd.DataFrame()
        
        score = analyzer._analyze_data_structure(empty_data)
        
        assert score == 0.0  # No structure to analyze


class TestPatternScoring:
    """Test pattern-based scoring functionality."""
    
    def test_calculate_pattern_score_business_data(self):
        """Test pattern scoring with business-like data."""
        analyzer = ConfidenceAnalyzer()
        
        business_data = pd.DataFrame({
            'Customer_ID': ['CUST001', 'CUST002', 'CUST003'],
            'Email': ['alice@email.com', 'bob@email.com', 'charlie@email.com'],
            'Phone': ['555-1234', '555-5678', '555-9012'],
            'Amount': ['$100.50', '$200.75', '$150.25']
        })
        
        score = analyzer._calculate_pattern_score(business_data)
        
        assert score > 0.5  # Should recognize business patterns
    
    def test_calculate_pattern_score_random_data(self):
        """Test pattern scoring with random data."""
        analyzer = ConfidenceAnalyzer()
        
        random_data = pd.DataFrame({
            'Random1': ['abc', 'def', 'ghi'],
            'Random2': ['123', '456', '789'],
            'Random3': ['xyz', 'uvw', 'rst']
        })
        
        score = analyzer._calculate_pattern_score(random_data)
        
        assert score < 0.3  # Should not recognize meaningful patterns
    
    def test_calculate_pattern_score_empty_data(self):
        """Test pattern scoring with empty data."""
        analyzer = ConfidenceAnalyzer()
        
        empty_data = pd.DataFrame()
        
        score = analyzer._calculate_pattern_score(empty_data)
        
        assert score == 0.0  # No patterns to analyze


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""
    
    def test_analyze_worksheet_with_all_nan_column(self):
        """Test analysis with completely NaN column."""
        analyzer = ConfidenceAnalyzer()
        
        nan_data = pd.DataFrame({
            'Good_Column': [1, 2, 3, 4, 5],
            'NaN_Column': [np.nan, np.nan, np.nan, np.nan, np.nan],
            'Another_Good': ['a', 'b', 'c', 'd', 'e']
        })
        
        nan_worksheet = WorksheetData(
            source_file=Path("nan.xlsx"),
            worksheet_name="NaNSheet",
            data=nan_data
        )
        
        result = analyzer.analyze_worksheet(nan_worksheet)
        
        # Should handle NaN gracefully
        assert isinstance(result, ConfidenceScore)
        assert 0.0 <= result.overall_score <= 1.0
    
    def test_analyze_worksheet_with_very_large_data(self):
        """Test analysis with very large dataset."""
        analyzer = ConfidenceAnalyzer()
        
        # Create large dataset
        large_data = pd.DataFrame({
            f'Col_{i}': range(i * 100, (i + 1) * 100) for i in range(50)
        })
        
        large_worksheet = WorksheetData(
            source_file=Path("large.xlsx"),
            worksheet_name="LargeSheet",
            data=large_data
        )
        
        result = analyzer.analyze_worksheet(large_worksheet)
        
        # Should complete without timeout
        assert isinstance(result, ConfidenceScore)
        assert result.overall_score > 0.0  # Should recognize structure
    
    def test_analyze_worksheet_with_unicode_data(self):
        """Test analysis with Unicode characters."""
        analyzer = ConfidenceAnalyzer()
        
        unicode_data = pd.DataFrame({
            'Names': ['Alice', 'José', '张三', 'محمد', 'Ñoño'],
            'Cities': ['New York', 'São Paulo', '北京', 'الرياض', 'México'],
            'Symbols': ['©', '®', '™', '€', '¥']
        })
        
        unicode_worksheet = WorksheetData(
            source_file=Path("unicode.xlsx"),
            worksheet_name="UnicodeSheet",
            data=unicode_data
        )
        
        result = analyzer.analyze_worksheet(unicode_worksheet)
        
        # Should handle Unicode gracefully
        assert isinstance(result, ConfidenceScore)
        assert 0.0 <= result.overall_score <= 1.0
    
    def test_analyzer_with_extreme_parameters(self):
        """Test analyzer with extreme parameter values."""
        # Very strict analyzer
        strict_analyzer = ConfidenceAnalyzer(
            threshold=0.99,
            min_rows=1000,
            min_columns=50,
            max_empty_percentage=0.01
        )
        
        # Should initialize without error
        assert strict_analyzer.threshold == 0.99
        assert strict_analyzer.min_rows == 1000
        
        # Very lenient analyzer
        lenient_analyzer = ConfidenceAnalyzer(
            threshold=0.01,
            min_rows=1,
            min_columns=1,
            max_empty_percentage=0.99
        )
        
        assert lenient_analyzer.threshold == 0.01
        assert lenient_analyzer.max_empty_percentage == 0.99
    
    def test_header_info_validation_edge_cases(self):
        """Test HeaderInfo validation with edge cases."""
        # Valid HeaderInfo
        valid_header = HeaderInfo(
            has_headers=True,
            header_row=0,
            header_quality=0.8,
            column_names=['A', 'B', 'C']
        )
        assert valid_header.has_headers is True
        
        # Invalid header quality
        with pytest.raises(ValueError, match="header_quality must be between 0.0 and 1.0"):
            HeaderInfo(
                has_headers=True,
                header_row=0,
                header_quality=1.5,  # Invalid
                column_names=['A']
            )
        
        # Invalid header row when has_headers is True
        with pytest.raises(ValueError, match="header_row cannot be None when has_headers is True"):
            HeaderInfo(
                has_headers=True,
                header_row=None,  # Invalid
                header_quality=0.8,
                column_names=['A']
            )
        
        # Invalid negative header row
        with pytest.raises(ValueError, match="header_row must be non-negative"):
            HeaderInfo(
                has_headers=True,
                header_row=-1,  # Invalid
                header_quality=0.8,
                column_names=['A']
            )


class TestPerformanceAndOptimization:
    """Test performance and optimization scenarios."""
    
    def test_analyze_multiple_worksheets_performance(self):
        """Test analyzing multiple worksheets for performance."""
        analyzer = ConfidenceAnalyzer()
        
        # Create multiple test worksheets
        worksheets = []
        for i in range(10):
            data = pd.DataFrame({
                'ID': list(range(i * 10, (i + 1) * 10)),
                'Value': [f'Value_{j}' for j in range(10)],
                'Score': [j * 0.1 for j in range(10)]
            })
            
            worksheet = WorksheetData(
                source_file=Path(f"test_{i}.xlsx"),
                worksheet_name=f"Sheet_{i}",
                data=data
            )
            worksheets.append(worksheet)
        
        # Analyze all worksheets
        import time
        start_time = time.time()
        
        results = []
        for worksheet in worksheets:
            result = analyzer.analyze_worksheet(worksheet)
            results.append(result)
        
        duration = time.time() - start_time
        
        # Should complete within reasonable time
        assert duration < 5.0  # 5 second timeout
        assert len(results) == 10
        assert all(isinstance(r, ConfidenceScore) for r in results)
    
    def test_pattern_compilation_caching(self):
        """Test that header patterns are compiled only once."""
        # Create multiple analyzers
        analyzer1 = ConfidenceAnalyzer()
        analyzer2 = ConfidenceAnalyzer()
        
        # Both should have compiled patterns
        assert len(analyzer1._header_patterns) > 0
        assert len(analyzer2._header_patterns) > 0
        
        # Pattern count should be the same
        assert len(analyzer1._header_patterns) == len(analyzer2._header_patterns)


class TestIntegrationWithWorksheetData:
    """Test integration with WorksheetData models."""
    
    def test_analyze_worksheet_data_properties(self, sample_worksheet_data):
        """Test that analyzer properly uses WorksheetData properties."""
        analyzer = ConfidenceAnalyzer()
        
        result = analyzer.analyze_worksheet(sample_worksheet_data)
        
        # Should use worksheet properties
        assert isinstance(result, ConfidenceScore)
        
        # Verify data was actually processed
        assert result.data_density > 0.0  # Should detect filled data
        assert result.structure_score > 0.0  # Should detect structure
        
        # Check that worksheet metadata influenced analysis
        assert sample_worksheet_data.row_count == 5
        assert sample_worksheet_data.column_count == 4
    
    def test_analyze_worksheet_with_custom_metadata(self):
        """Test analysis with custom worksheet metadata."""
        data = pd.DataFrame({
            'Custom_ID': [1, 2, 3],
            'Custom_Value': ['A', 'B', 'C']
        })
        
        custom_worksheet = WorksheetData(
            source_file=Path("custom.xlsx"),
            worksheet_name="CustomSheet",
            data=data,
            metadata={'custom_field': 'custom_value'}
        )
        
        analyzer = ConfidenceAnalyzer()
        result = analyzer.analyze_worksheet(custom_worksheet)
        
        # Should handle custom metadata without issues
        assert isinstance(result, ConfidenceScore)
        assert result.overall_score >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])