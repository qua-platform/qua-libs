from typing import Optional

from pydantic import Field
from qualibrate.parameters import RunnableParameters

class DataLoadableNodeParameters(RunnableParameters):
    load_data_id: Optional[int] = Field(None, description="QUAlibrate node run index for loading historical data.")


