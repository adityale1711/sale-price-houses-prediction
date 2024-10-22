from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.preprocessing import Binarizer, MinMaxScaler
from Production.processing import features as pp
from Production.config.core import config
from feature_engine.wrappers import SklearnTransformerWrapper
from feature_engine.encoding import RareLabelEncoder, OrdinalEncoder
from feature_engine.selection import DropFeatures
from feature_engine.imputation import CategoricalImputer, AddMissingIndicator, MeanMedianImputer
from feature_engine.transformation import LogTransformer

price_pipe = Pipeline(
    [
        # impute categorical variables with string missing
        (
            'missing_imputation',
            CategoricalImputer(
                imputation_method='missing',
                variables=config.model_config.categorical_vars_with_na_missing
            )
        ),
        (
            'frequent_imputation',
            CategoricalImputer(
                imputation_method='frequent',
                variables=config.model_config.categorical_vars_with_na_frequent
            )
        ),

        # Add Missing Indicator
        (
            'missing_indicator',
            AddMissingIndicator(variables=config.model_config.numerical_vars_with_na)
        ),

        # Impute numerical variables with the mean
        (
            'mean_imputation',
            MeanMedianImputer(config.model_config.numerical_vars_with_na)
        ),

        # Temporal variables
        (
            'elapsed_time',
            pp.TemporalVariableTransformer(
                variables=config.model_config.temporal_vars,
                reference_variable=config.model_config.ref_var
            )
        ),
        (
            'drop_features',
            DropFeatures(features_to_drop=[config.model_config.ref_var])
        ),

        # Variable Transformation
        (
            'log',
            LogTransformer(variables=config.model_config.numerical_log_vars)
        ),
        (
            'binarizer',
            SklearnTransformerWrapper(transformer=Binarizer(threshold=0), variables=config.model_config.binarize_vars)
        ),

        # Mappers
        (
            'mapper_qual',
            pp.Mapper(variables=config.model_config.qual_vars, mappings=config.model_config.qual_mappings)
        ),
        (
            'mapper_exposure',
            pp.Mapper(variables=config.model_config.exposure_vars, mappings=config.model_config.exposure_mappings)
        ),
        (
            'mapper_finish',
            pp.Mapper(variables=config.model_config.finish_vars, mappings=config.model_config.finish_mappings)
        ),
        (
            'mapper_garage',
            pp.Mapper(variables=config.model_config.garage_vars, mappings=config.model_config.garage_mappings)
        ),

        # Categorical Encoding
        (
            'rare_label_encoder',
            RareLabelEncoder(tol=0.01, n_categories=1, variables=config.model_config.categorical_vars)
        ),

        # Encode categorical variables using the target mean
        (
            'categorical_encoder',
            OrdinalEncoder(encoding_method='ordered', variables=config.model_config.categorical_vars)
        ),
        (
            'scaler',
            MinMaxScaler()
        ),
        (
            'Lasso',
            Lasso(alpha=config.model_config.alpha, random_state=config.model_config.random_state)
        )
    ]
)