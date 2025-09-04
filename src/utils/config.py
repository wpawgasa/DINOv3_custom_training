from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ValidationError


class ConfigValidationError(Exception):
    pass


class ConfigLoader:
    def __init__(self):
        self._loaded_configs: Dict[str, DictConfig] = {}
    
    def load_config(self, config_path: Union[str, Path]) -> DictConfig:
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        config_key = str(config_path.absolute())
        if config_key in self._loaded_configs:
            return self._loaded_configs[config_key]
        
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config_dict = yaml.safe_load(file)
            
            if config_dict is None:
                config_dict = {}
            
            config = OmegaConf.create(config_dict)
            self._loaded_configs[config_key] = config
            return config
            
        except yaml.YAMLError as e:
            raise ConfigValidationError(f"Invalid YAML in {config_path}: {e}")
        except Exception as e:
            raise ConfigValidationError(f"Error loading config {config_path}: {e}")
    
    def merge_configs(self, *configs: Union[str, Path, DictConfig, Dict]) -> DictConfig:
        merged_config = OmegaConf.create({})
        
        for config in configs:
            if isinstance(config, (str, Path)):
                config = self.load_config(config)
            elif isinstance(config, dict):
                config = OmegaConf.create(config)
            elif not isinstance(config, DictConfig):
                raise ValueError(f"Unsupported config type: {type(config)}")
            
            merged_config = OmegaConf.merge(merged_config, config)
        
        return merged_config
    
    def apply_overrides(self, config: DictConfig, overrides: List[str]) -> DictConfig:
        config_copy = OmegaConf.create(config)
        
        for override in overrides:
            try:
                key, value = override.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                try:
                    parsed_value = yaml.safe_load(value)
                except yaml.YAMLError:
                    parsed_value = value
                
                OmegaConf.set(config_copy, key, parsed_value)
                
            except ValueError:
                raise ConfigValidationError(f"Invalid override format: {override}. Expected 'key=value'")
        
        return config_copy
    
    def validate_config(self, config: DictConfig, schema_class: Optional[BaseModel] = None) -> bool:
        if schema_class is None:
            return True
        
        try:
            config_dict = OmegaConf.to_container(config, resolve=True)
            schema_class(**config_dict)
            return True
        except ValidationError as e:
            raise ConfigValidationError(f"Configuration validation failed: {e}")
        except Exception as e:
            raise ConfigValidationError(f"Unexpected validation error: {e}")
    
    def save_config(self, config: DictConfig, output_path: Union[str, Path]) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            config_dict = OmegaConf.to_container(config, resolve=True)
            with open(output_path, 'w', encoding='utf-8') as file:
                yaml.dump(config_dict, file, default_flow_style=False, indent=2)
        except Exception as e:
            raise ConfigValidationError(f"Error saving config to {output_path}: {e}")
    
    def get_config_value(self, config: DictConfig, key: str, default: Any = None) -> Any:
        try:
            return OmegaConf.select(config, key, default=default)
        except Exception:
            return default
    
    def clear_cache(self) -> None:
        self._loaded_configs.clear()


def load_hierarchical_config(
    base_config: Union[str, Path],
    model_config: Optional[Union[str, Path]] = None,
    training_config: Optional[Union[str, Path]] = None,
    dataset_config: Optional[Union[str, Path]] = None,
    overrides: Optional[List[str]] = None
) -> DictConfig:
    loader = ConfigLoader()
    configs_to_merge = []
    
    configs_to_merge.append(base_config)
    
    if model_config:
        configs_to_merge.append(model_config)
    
    if training_config:
        configs_to_merge.append(training_config)
    
    if dataset_config:
        configs_to_merge.append(dataset_config)
    
    merged_config = loader.merge_configs(*configs_to_merge)
    
    if overrides:
        merged_config = loader.apply_overrides(merged_config, overrides)
    
    return merged_config


_default_loader = ConfigLoader()

load_config = _default_loader.load_config
merge_configs = _default_loader.merge_configs
apply_overrides = _default_loader.apply_overrides
validate_config = _default_loader.validate_config
save_config = _default_loader.save_config
get_config_value = _default_loader.get_config_value
clear_config_cache = _default_loader.clear_cache