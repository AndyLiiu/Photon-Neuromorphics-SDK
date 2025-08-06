"""
Component registry for custom photonic components.
"""

from typing import Dict, Type, Any
import inspect


class ComponentRegistry:
    """Registry for photonic components."""
    
    _instance = None
    _components: Dict[str, Type] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(cls, name: str, component_class: Type):
        """Register a component class."""
        if not hasattr(component_class, 'forward'):
            raise ValueError(f"Component {name} must implement forward() method")
        if not hasattr(component_class, 'to_netlist'):
            raise ValueError(f"Component {name} must implement to_netlist() method")
            
        cls._components[name] = component_class
        
    @classmethod
    def get(cls, name: str) -> Type:
        """Get a registered component class."""
        if name not in cls._components:
            raise KeyError(f"Component {name} not registered")
        return cls._components[name]
        
    @classmethod
    def list_components(cls) -> Dict[str, Type]:
        """List all registered components."""
        return cls._components.copy()
        
    @classmethod
    def create(cls, name: str, **kwargs) -> Any:
        """Create an instance of a registered component."""
        component_class = cls.get(name)
        return component_class(**kwargs)


def register_component(name_or_class=None):
    """Decorator to register a photonic component.
    
    Usage:
        @register_component
        class MyComponent(PhotonicComponent):
            ...
            
        @register_component("custom_name")
        class MyComponent(PhotonicComponent):
            ...
    """
    def decorator(cls):
        component_name = name_or_class if isinstance(name_or_class, str) else cls.__name__
        ComponentRegistry.register(component_name, cls)
        return cls
        
    if inspect.isclass(name_or_class):
        # Used without parentheses: @register_component
        return decorator(name_or_class)
    else:
        # Used with name: @register_component("name")
        return decorator


# Global registry instance
registry = ComponentRegistry()