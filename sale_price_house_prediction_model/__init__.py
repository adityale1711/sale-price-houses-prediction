import logging
from sale_price_house_prediction_model.config.core import config, PACKAGE_ROOT

logging.getLogger(config.app_config.package_name).addHandler(logging.NullHandler())
with open(PACKAGE_ROOT / 'VERSION') as version_file:
    __version__ = version_file.read().strip()