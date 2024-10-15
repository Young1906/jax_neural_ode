from pydantic import BaseModel
from pydantic import PositiveFloat
from pydantic import PositiveInt


class Config(BaseModel):
    c: PositiveFloat  # dampener coefficient
    k: PositiveFloat  # spring coef
    m: PositiveFloat  # mass
    A: float  # Initial position
    V: float  # Initial velocity
    t0: float  # Start time
    t1: float  # Terminal time
    img_pth: str  # Path to save figure


class DatasetConfig(BaseModel):
    tmin                : float 
    tmax                : float 
    n_samples           : PositiveInt
    batch_size          : PositiveInt       # batch's size
    seq_len             : PositiveInt
    noise_scale         : float 

class TrainingConfig(BaseModel):
    n_iter              : PositiveInt 
    learning_rate       : PositiveFloat

class SolverConfig(BaseModel):
    h_max               : PositiveFloat

class ModelConfig(BaseModel):
    u_dim               : PositiveFloat
    mlp_layers          : list[PositiveInt]


class NeuralODEConfig(BaseModel):
    seed                : int               # main program's random seed
    dataset             : DatasetConfig
    model               : ModelConfig
    solver              : SolverConfig
    training            : TrainingConfig
