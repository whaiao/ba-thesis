from typing import Mapping
import transformers
import torch.nn as nn

ModelTypeDict = Mapping[str, transformers.PreTrainedModel]
ModelConfigDict = Mapping[str, transformers.AutoConfig]
TaskModelDict = Mapping[str, transformers.PreTrainedModel]


class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder: nn.Module, taskmodel_dict: TaskModelDict):
        super().__init__(transformers.configuration_utils.PretrainedConfig())
        self.encoder = encoder
        self.taskmodel_dict = nn.ModuleDict(taskmodel_dict)

    @classmethod
    def create(cls, model_name: str, model_type_dict: ModelTypeDict, model_config_dict: ModelConfigDict):
        shared_encoder = None
        taskmodel_dict = {}
        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained(
                model_name,
                config=model_config_dict[task_name],
            )
            if shared_encoder is None:
                shared_encoder = getattr(model, cls.get_encoder_attr_name(model))
            else:
                setattr(model, cls.get_encoder_attr_name(model), shared_encoder)

            taskmodel_dict[task_name] = model
        return cls(encoder=shared_encoder, taskmodel_dict=taskmodel_dict)

    @staticmethod
    def get_encoder_attr_name(model):
        model_class_name = model.__class__.__name__
        if model_class_name.startswith('Bert'):
            return 'bert'
        elif model_class_name.startswith('Roberta'):
            return 'roberta'
        elif model_class_name.startswith('Albert'):
            return 'albert'
        elif model_class_name.startswith('Long'):  # Longformer uses RoBERTa as a base class
            return 'roberta'
        else:
            raise KeyError(f'Add support for new model {model_class_name}')
