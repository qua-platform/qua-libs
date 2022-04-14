def remove_delays(config):
    for k, v in config.items():

        if "elements" in config:  # Only look at elements
            config["elements"] = remove_delays(config["elements"])
            return config

        if isinstance(v, dict):  # Search recursively
            config[k] = remove_delays(v)

    if "delay" in config:
        config["delay"] = 0
    if "buffer" in config:
        config["buffer"] = 0

    return config
