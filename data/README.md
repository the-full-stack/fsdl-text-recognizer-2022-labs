# `data` folder overview

Any data that needs to be stored locally should be saved in this location. This folder,
and its sub-folders, are not version-controlled.

The sub-folders should be used as follows:

  - `external`: any data that will not be processed at all, such as reference data;
  - `raw`: any raw data before any processing;
  - `interim`: any raw data that has been partially processed and, for whatever reason,
    needs to be stored before further processing is completed; and
  - `processed`: any raw or interim data that has been fully processed into its final
    state.
