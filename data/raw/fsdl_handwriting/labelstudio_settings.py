# first bring in basic settings
from label_studio.core.settings.base import *

# edit those settings: skip version check to reduce terminal noise
LATEST_VERSION_CHECK = False

# then bring in top-level settings
#   this is where the noisy version check happens
from label_studio.core.settings.label_studio import *
