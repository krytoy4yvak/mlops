from dataclasses import dataclass, field
from typing import List, Optional


@dataclass()
class FeatureParams:
    # categorical_features: List[str]
    numerical_features: List[str]
    # features_to_drop: List[str]
    target_col: Optional[str]
    need_proba: bool = field(default=True)
