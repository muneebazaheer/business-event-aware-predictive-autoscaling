import numpy as np
import pandas as pd
from pathlib import Path
import logging
import pickle
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from typing import Dict, Tuple, Optional
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkloadClassifierTrainer:
    """Train and evaluate workload archetype classifiers."""
    
    def __init__(self, output_dir: str = "models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def prepare_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple:
        """Prepare data for training."""
        # Define feature columns (exclude metadata and labels)
        metadata_cols = ['function_hash', 'window_idx', 'window_start_minute', 
                        'window_end_minute', 'day', 'hour_of_day', 'label', 'confidence']
        feature_cols = [col for col in train_df.columns if col not in metadata_cols]
        
        # Extract features and labels
        X_train = train_df[feature_cols].values
        y_train = train_df['label'].values
        X_val = val_df[feature_cols].values
        y_val = val_df['label'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['label'].values
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        self.feature_names = feature_cols
        
        logger.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        logger.info(f"Features: {len(feature_cols)}")
        
        return (X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test)
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val, use_gpu: bool = True):
        """Train LightGBM classifier."""
        logger.info("Training LightGBM classifier...")
        
        # Encode labels
        label_map = {label: i for i, label in enumerate(np.unique(y_train))}
        reverse_label_map = {i: label for label, i in label_map.items()}
        
        y_train_encoded = np.array([label_map[label] for label in y_train])
        y_val_encoded = np.array([label_map[label] for label in y_val])
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train_encoded)
        val_data = lgb.Dataset(X_val, label=y_val_encoded, reference=train_data)
        
        # Parameters
        params = {
            'objective': 'multiclass',
            'num_class': len(label_map),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'num_threads': -1,
        }
        
        if use_gpu:
            params['device'] = 'gpu'
            params['gpu_platform_id'] = 0
            params['gpu_device_id'] = 0
            logger.info("Using GPU acceleration for LightGBM")
        
        # Train
        start_time = time.time()
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=300,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
        )
        training_time = time.time() - start_time
        logger.info(f"LightGBM training completed in {training_time:.2f} seconds")
        
        # Store model and metadata
        self.models['lightgbm'] = model
        self.models['lightgbm_label_map'] = label_map
        self.models['lightgbm_reverse_map'] = reverse_label_map
        
        # Feature importance
        importance = model.feature_importance(importance_type='gain')
        self.feature_importance['lightgbm'] = dict(zip(self.feature_names, importance))
        
        return model
    
    def train_xgboost(self, X_train, y_train, X_val, y_val, use_gpu: bool = True):
        """Train XGBoost classifier."""
        logger.info("Training XGBoost classifier...")
        
        # Encode labels
        label_map = {label: i for i, label in enumerate(np.unique(y_train))}
        reverse_label_map = {i: label for label, i in label_map.items()}
        
        y_train_encoded = np.array([label_map[label] for label in y_train])
        y_val_encoded = np.array([label_map[label] for label in y_val])
        
        # Parameters
        params = {
            'objective': 'multi:softprob',
            'num_class': len(label_map),
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 50,
            'eval_metric': 'mlogloss'
        }
        
        if use_gpu:
            params['tree_method'] = 'gpu_hist'
            params['predictor'] = 'gpu_predictor'
            logger.info("Using GPU acceleration for XGBoost")
        
        # Train
        start_time = time.time()
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train_encoded,
            eval_set=[(X_val, y_val_encoded)],
            verbose=True
        )
        training_time = time.time() - start_time
        logger.info(f"XGBoost training completed in {training_time:.2f} seconds")
        
        # Store model and metadata
        self.models['xgboost'] = model
        self.models['xgboost_label_map'] = label_map
        self.models['xgboost_reverse_map'] = reverse_label_map
        
        return model
    
    def evaluate_model(self, model_name: str, X_test, y_test) -> Dict:
        """Evaluate a trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        model = self.models[model_name]
        label_map = self.models[f'{model_name}_label_map']
        reverse_map = self.models[f'{model_name}_reverse_map']
        
        # Encode test labels
        y_test_encoded = np.array([label_map[label] for label in y_test])
        
        # Predict
        if model_name == 'lightgbm':
            y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
            y_pred_encoded = np.argmax(y_pred_proba, axis=1)
        else:  # xgboost
            y_pred_proba = model.predict_proba(X_test)
            y_pred_encoded = model.predict(X_test)
        
        # Decode predictions
        y_pred = np.array([reverse_map[pred] for pred in y_pred_encoded])
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate confidence scores
        max_proba = np.max(y_pred_proba, axis=1)
        avg_confidence = np.mean(max_proba)
        
        results = {
            'classification_report': report,
            'confusion_matrix': cm,
            'avg_confidence': avg_confidence,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'labels': list(reverse_map.values())
        }
        
        return results
    
    def plot_confusion_matrix(self, results: Dict, model_name: str, save_path: Optional[str] = None):
        """Plot confusion matrix."""
        cm = results['confusion_matrix']
        labels = results['labels']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self, model_name: str, top_n: int = 20, save_path: Optional[str] = None):
        """Plot feature importance."""
        if model_name not in self.feature_importance:
            logger.warning(f"No feature importance for {model_name}")
            return
            
        importance_dict = self.feature_importance[model_name]
        
        # Sort and get top features
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, importances = zip(*sorted_features)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Features - {model_name}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_models(self):
        """Save trained models and scalers."""
        # Save models
        for name, model in self.models.items():
            if 'map' not in name:  # Skip label maps
                model_path = self.output_dir / f"{name}_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Saved {name} model to {model_path}")
        
        # Save scalers
        scaler_path = self.output_dir / "scalers.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scalers, f)
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'label_maps': {
                name: self.models[name] 
                for name in self.models if 'map' in name
            },
            'feature_importance': self.feature_importance
        }
        
        metadata_path = self.output_dir / "model_metadata.yaml"
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f)
        
        logger.info(f"Saved model metadata to {metadata_path}")


def train_all_models(data_dir: str = "data_production", output_dir: str = "models"):
    """Main training pipeline."""
    logger.info("Starting model training pipeline...")
    
    # Load data
    train_df = pd.read_pickle(Path(data_dir) / "train_windows.pkl")
    val_df = pd.read_pickle(Path(data_dir) / "val_windows.pkl")
    test_df = pd.read_pickle(Path(data_dir) / "test_windows.pkl")
    
    logger.info(f"Loaded data - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Initialize trainer
    trainer = WorkloadClassifierTrainer(output_dir)
    
    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = trainer.prepare_data(train_df, val_df, test_df)
    
    # Train models
    trainer.train_lightgbm(X_train, y_train, X_val, y_val, use_gpu=True)
    trainer.train_xgboost(X_train, y_train, X_val, y_val, use_gpu=True)
    
    # Evaluate models
    results = {}
    for model_name in ['lightgbm', 'xgboost']:
        logger.info(f"\nEvaluating {model_name}...")
        results[model_name] = trainer.evaluate_model(model_name, X_test, y_test)
        
        # Print metrics
        report = results[model_name]['classification_report']
        print(f"\n{model_name.upper()} Results:")
        print(f"Accuracy: {report['accuracy']:.3f}")
        print(f"Avg Confidence: {results[model_name]['avg_confidence']:.3f}")
        print("\nPer-class metrics:")
        for label in results[model_name]['labels']:
            metrics = report[label]
            print(f"  {label}: precision={metrics['precision']:.3f}, "
                  f"recall={metrics['recall']:.3f}, f1={metrics['f1-score']:.3f}")
        
        # Plot confusion matrix
        trainer.plot_confusion_matrix(
            results[model_name], 
            model_name,
            save_path=Path(output_dir) / f"confusion_matrix_{model_name}.png"
        )
        
        # Plot feature importance
        trainer.plot_feature_importance(
            model_name,
            top_n=15,
            save_path=Path(output_dir) / f"feature_importance_{model_name}.png"
        )
    
    # Save models
    trainer.save_models()
    
    logger.info("\nModel training completed!")
    return trainer, results


if __name__ == "__main__":
    trainer, results = train_all_models()