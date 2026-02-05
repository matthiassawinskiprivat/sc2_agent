class ProtoWrapper:
    """
    A helper class to wrap SC2 protocol objects.
    It ensures that attributes can be accessed even if the library
    expects a slightly different structure (e.g. ResponseGameInfo vs GameInfo).
    """
    def __init__(self, proto):
        self._proto = proto

    def __getattr__(self, name):
        # Handle library expecting a Response object when we have a ResponseGameInfo
        if name == "game_info" and not hasattr(self._proto, "game_info"):
            return self._proto

        # First try to get it from self._proto directly
        try:
            return getattr(self._proto, name)
        except AttributeError:
            # If not found, try to get it from self._proto.game_info if it exists
            # This handles cases where we wrapped a ResponseGameInfo but 
            # library code expect it to act like GameInfo
            if hasattr(self._proto, "game_info"):
                return getattr(self._proto.game_info, name)
            raise AttributeError(f"'{type(self._proto).__name__}' has no attribute '{name}'")
