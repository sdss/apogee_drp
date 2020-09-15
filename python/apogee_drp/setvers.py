# set up a global file loader for all modules with a default version

# The version/telescope can then be modified by any routine as needed

from apogee_drp.utils import apload
load=apload.ApLoad()
