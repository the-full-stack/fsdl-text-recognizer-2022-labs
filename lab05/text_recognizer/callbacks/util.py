from loguru import logger as log


def check_and_warn(logger, attribute, feature):
    if not hasattr(logger, attribute):
        warn_no_attribute(feature, attribute)
        return True
    return False


def warn_no_attribute(blocked_feature, missing_attribute):
    log.warning(
        f"Unable to log {blocked_feature}: logger does not have attribute {missing_attribute}.",
    )
