# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Match Cambricon device types, including renamed PrivateUse1 backends."""

import torch

from flag_gems.runtime import device


def is_backend_device_type(device_type: str) -> bool:
    """The vendor name (mlu) and PyTorch's registered name (e.g. flagos) differ.

    Query the registration at call time: torch-fl can register it after import.
    """
    if device_type == device.name:
        return True
    get_name = getattr(torch._C, "_get_privateuse1_backend_name", None)
    return get_name is not None and device_type == get_name()
