"""
Data serialization utilities.
Handles serialization/deserialization of trading data for storage and transmission.
"""

import logging
import json
import pickle
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, date, time
from decimal import Decimal
from enum import Enum
import base64

import numpy as np
import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for trading data types."""

    def default(self, obj):
        """
        Encode custom types.

        Args:
            obj: Object to encode

        Returns:
            Encoded object
        """
        # DateTime types
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, date):
            return obj.isoformat()
        elif isinstance(obj, time):
            return obj.isoformat()

        # Numeric types
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        # Pandas types
        elif isinstance(obj, pd.DataFrame):
            return {
                "__type__": "DataFrame",
                "data": obj.to_dict(orient="records"),
                "index": obj.index.tolist(),
                "columns": obj.columns.tolist()
            }
        elif isinstance(obj, pd.Series):
            return {
                "__type__": "Series",
                "data": obj.to_dict(),
                "index": obj.index.tolist(),
                "name": obj.name
            }

        # Enum types
        elif isinstance(obj, Enum):
            return obj.value

        # Pydantic models
        elif isinstance(obj, BaseModel):
            return obj.model_dump()

        # Sets
        elif isinstance(obj, set):
            return list(obj)

        # Bytes
        elif isinstance(obj, bytes):
            return base64.b64encode(obj).decode('utf-8')

        # Complex numbers
        elif isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}

        return super().default(obj)


class CustomJSONDecoder(json.JSONDecoder):
    """Custom JSON decoder for trading data types."""

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        """
        Decode custom types.

        Args:
            obj: Object to decode

        Returns:
            Decoded object
        """
        if "__type__" not in obj:
            return obj

        obj_type = obj["__type__"]

        if obj_type == "DataFrame":
            df = pd.DataFrame(obj["data"])
            df.index = obj.get("index", df.index)
            df.columns = obj.get("columns", df.columns)
            return df

        elif obj_type == "Series":
            series = pd.Series(obj["data"])
            series.index = obj.get("index", series.index)
            series.name = obj.get("name")
            return series

        return obj


def serialize_json(
    data: Any,
    pretty: bool = False
) -> str:
    """
    Serialize data to JSON string.

    Args:
        data: Data to serialize
        pretty: Whether to pretty-print

    Returns:
        JSON string
    """
    indent = 2 if pretty else None
    return json.dumps(data, cls=CustomJSONEncoder, indent=indent)


def deserialize_json(json_str: str) -> Any:
    """
    Deserialize JSON string to data.

    Args:
        json_str: JSON string

    Returns:
        Deserialized data
    """
    return json.loads(json_str, cls=CustomJSONDecoder)


def serialize_pickle(data: Any) -> bytes:
    """
    Serialize data using pickle.

    Args:
        data: Data to serialize

    Returns:
        Pickled bytes
    """
    return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_pickle(data: bytes) -> Any:
    """
    Deserialize pickle data.

    Args:
        data: Pickled bytes

    Returns:
        Deserialized data
    """
    return pickle.loads(data)


def serialize_dataframe(
    df: pd.DataFrame,
    format: str = "parquet"
) -> bytes:
    """
    Serialize DataFrame to bytes.

    Args:
        df: DataFrame to serialize
        format: Serialization format ("parquet", "feather", "pickle")

    Returns:
        Serialized bytes
    """
    if format == "parquet":
        return df.to_parquet(compression="snappy")
    elif format == "feather":
        return df.to_feather(compression="lz4")
    elif format == "pickle":
        return pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError(f"Unknown format: {format}")


def deserialize_dataframe(
    data: bytes,
    format: str = "parquet"
) -> pd.DataFrame:
    """
    Deserialize DataFrame from bytes.

    Args:
        data: Serialized bytes
        format: Serialization format ("parquet", "feather", "pickle")

    Returns:
        Deserialized DataFrame
    """
    if format == "parquet":
        return pd.read_parquet(data)
    elif format == "feather":
        return pd.read_feather(data)
    elif format == "pickle":
        return pickle.loads(data)
    else:
        raise ValueError(f"Unknown format: {format}")


class OrderSerializer:
    """Serializes/deserializes order data."""

    @staticmethod
    def serialize(order: Dict[str, Any]) -> str:
        """
        Serialize order to JSON.

        Args:
            order: Order dictionary

        Returns:
            JSON string
        """
        return serialize_json(order)

    @staticmethod
    def deserialize(json_str: str) -> Dict[str, Any]:
        """
        Deserialize order from JSON.

        Args:
            json_str: JSON string

        Returns:
            Order dictionary
        """
        return deserialize_json(json_str)

    @staticmethod
    def to_bytes(order: Dict[str, Any]) -> bytes:
        """
        Convert order to bytes.

        Args:
            order: Order dictionary

        Returns:
            Bytes representation
        """
        return serialize_json(order).encode("utf-8")

    @staticmethod
    def from_bytes(data: bytes) -> Dict[str, Any]:
        """
        Parse order from bytes.

        Args:
            data: Bytes data

        Returns:
            Order dictionary
        """
        return deserialize_json(data.decode("utf-8"))


class PositionSerializer:
    """Serializes/deserializes position data."""

    @staticmethod
    def serialize(position: Dict[str, Any]) -> str:
        """
        Serialize position to JSON.

        Args:
            position: Position dictionary

        Returns:
            JSON string
        """
        return serialize_json(position)

    @staticmethod
    def deserialize(json_str: str) -> Dict[str, Any]:
        """
        Deserialize position from JSON.

        Args:
            json_str: JSON string

        Returns:
            Position dictionary
        """
        return deserialize_json(json_str)


class MarketDataSerializer:
    """Serializes/deserializes market data."""

    @staticmethod
    def serialize_tick(tick: Dict[str, Any]) -> str:
        """
        Serialize tick data to JSON.

        Args:
            tick: Tick dictionary

        Returns:
            JSON string
        """
        return serialize_json(tick)

    @staticmethod
    def deserialize_tick(json_str: str) -> Dict[str, Any]:
        """
        Deserialize tick data from JSON.

        Args:
            json_str: JSON string

        Returns:
            Tick dictionary
        """
        return deserialize_json(json_str)

    @staticmethod
    def serialize_bars(bars: pd.DataFrame) -> bytes:
        """
        Serialize OHLCV bars to bytes.

        Args:
            bars: OHLCV DataFrame

        Returns:
            Serialized bytes
        """
        return serialize_dataframe(bars, format="parquet")

    @staticmethod
    def deserialize_bars(data: bytes) -> pd.DataFrame:
        """
        Deserialize OHLCV bars from bytes.

        Args:
            data: Serialized bytes

        Returns:
            OHLCV DataFrame
        """
        return deserialize_dataframe(data, format="parquet")


class ModelSerializer:
    """Serializes/deserializes model data."""

    @staticmethod
    def serialize_state(state: Dict[str, Any]) -> bytes:
        """
        Serialize model state.

        Args:
            state: Model state dictionary

        Returns:
            Serialized bytes
        """
        return serialize_pickle(state)

    @staticmethod
    def deserialize_state(data: bytes) -> Dict[str, Any]:
        """
        Deserialize model state.

        Args:
            data: Serialized bytes

        Returns:
            Model state dictionary
        """
        return deserialize_pickle(data)

    @staticmethod
    def serialize_predictions(
        predictions: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> bytes:
        """
        Serialize model predictions.

        Args:
            predictions: Predictions array/Series/DataFrame

        Returns:
            Serialized bytes
        """
        if isinstance(predictions, (pd.Series, pd.DataFrame)):
            return serialize_dataframe(predictions, format="parquet")
        else:
            return serialize_pickle(predictions)

    @staticmethod
    def deserialize_predictions(
        data: bytes,
        as_dataframe: bool = False
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Deserialize model predictions.

        Args:
            data: Serialized bytes
            as_dataframe: Whether to return as DataFrame

        Returns:
            Predictions
        """
        if as_dataframe:
            return deserialize_dataframe(data, format="parquet")
        else:
            return deserialize_pickle(data)


class MetricsSerializer:
    """Serializes/deserializes metrics data."""

    @staticmethod
    def serialize(metrics: Dict[str, Any]) -> str:
        """
        Serialize metrics to JSON.

        Args:
            metrics: Metrics dictionary

        Returns:
            JSON string
        """
        return serialize_json(metrics)

    @staticmethod
    def deserialize(json_str: str) -> Dict[str, Any]:
        """
        Deserialize metrics from JSON.

        Args:
            json_str: JSON string

        Returns:
            Metrics dictionary
        """
        return deserialize_json(json_str)

    @staticmethod
    def serialize_timeseries(
        metrics: pd.DataFrame
    ) -> bytes:
        """
        Serialize time series metrics.

        Args:
            metrics: Metrics DataFrame

        Returns:
            Serialized bytes
        """
        return serialize_dataframe(metrics, format="parquet")

    @staticmethod
    def deserialize_timeseries(data: bytes) -> pd.DataFrame:
        """
        Deserialize time series metrics.

        Args:
            data: Serialized bytes

        Returns:
            Metrics DataFrame
        """
        return deserialize_dataframe(data, format="parquet")


def compress_data(data: bytes, algorithm: str = "gzip") -> bytes:
    """
    Compress data.

    Args:
        data: Data to compress
        algorithm: Compression algorithm ("gzip", "lz4", "zstd")

    Returns:
        Compressed data
    """
    if algorithm == "gzip":
        import gzip
        return gzip.compress(data)
    elif algorithm == "lz4":
        import lz4.frame
        return lz4.frame.compress(data)
    elif algorithm == "zstd":
        import zstandard as zstd
        cctx = zstd.ZstdCompressor()
        return cctx.compress(data)
    else:
        raise ValueError(f"Unknown compression algorithm: {algorithm}")


def decompress_data(data: bytes, algorithm: str = "gzip") -> bytes:
    """
    Decompress data.

    Args:
        data: Compressed data
        algorithm: Compression algorithm ("gzip", "lz4", "zstd")

    Returns:
        Decompressed data
    """
    if algorithm == "gzip":
        import gzip
        return gzip.decompress(data)
    elif algorithm == "lz4":
        import lz4.frame
        return lz4.frame.decompress(data)
    elif algorithm == "zstd":
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data)
    else:
        raise ValueError(f"Unknown compression algorithm: {algorithm}")


def hash_data(data: Any) -> str:
    """
    Generate hash of data.

    Args:
        data: Data to hash

    Returns:
        Hash string
    """
    import hashlib

    # Serialize to bytes
    if isinstance(data, bytes):
        data_bytes = data
    elif isinstance(data, str):
        data_bytes = data.encode("utf-8")
    else:
        data_bytes = serialize_pickle(data)

    # Generate hash
    return hashlib.sha256(data_bytes).hexdigest()


def encode_base64(data: bytes) -> str:
    """
    Encode bytes to base64 string.

    Args:
        data: Bytes to encode

    Returns:
        Base64 string
    """
    return base64.b64encode(data).decode("utf-8")


def decode_base64(data: str) -> bytes:
    """
    Decode base64 string to bytes.

    Args:
        data: Base64 string

    Returns:
        Decoded bytes
    """
    return base64.b64decode(data.encode("utf-8"))
