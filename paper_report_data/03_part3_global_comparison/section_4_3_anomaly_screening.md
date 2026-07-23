### 4.3 Anomaly Screening and Candidate Identification

The anomaly screening stage was based on the master residual between the observed global polytropic structure and the amount of deviation explained by the five physical drivers. The anomaly score was defined as

\[
\delta_{global} = (n_{observed} - n_{base}) - \sum \langle \Delta n_i \rangle ,
\]

where \(n_{observed}\) is the fitted global polytropic index, \(n_{base}\) is the expected reference polytropic index, and \(\sum \langle \Delta n_i \rangle\) is the combined contribution from radiation pressure, composition gradients, convection, nuclear energy generation, and degeneracy. A large local feature is therefore not automatically an anomaly. For example, the superadiabatic surface layer can produce a strong convective spike, but this feature is classified as normal when \(\Delta n_{conv}\) accounts for it and leaves \(\delta_{global}\approx0\). In this implementation, a profile is screened as anomalous only when \(|\delta_{global}|>5.0\), indicating residual structure beyond the noise threshold of the current five-driver model.

| Star/Profile ID | Mass | Age | Global \(n\) | Anomaly Score \(\delta_{global}\) | Classification | Diagnostic Reason |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| mesa_model_0001_profile_01 | \(1.000\,M_\odot\) | 1.000e-14 Gyr | 1.923 | -0.594 | Normal | Deviation is accounted for by the convective driver \(\Delta n_{conv}=0.898\). The superadiabatic surface layer is therefore treated as explained structure, not an unresolved anomaly. |
| mesa_model_0050_profile_02 | \(1.000\,M_\odot\) | 4.550e-10 Gyr | 1.922 | -0.909 | Normal | Deviation is accounted for by the convective driver \(\Delta n_{conv}=1.216\). The superadiabatic surface layer is therefore treated as explained structure, not an unresolved anomaly. |
| mesa_model_0100_profile_03 | \(1.000\,M_\odot\) | 2.511e-06 Gyr | 1.800 | -1.045 | Normal | Deviation is accounted for by the convective driver \(\Delta n_{conv}=1.253\). The superadiabatic surface layer is therefore treated as explained structure, not an unresolved anomaly. |
| mesa_model_0150_profile_04 | \(1.000\,M_\odot\) | 1.107e-04 Gyr | 1.640 | -1.100 | Normal | Deviation is accounted for by the convective driver \(\Delta n_{conv}=1.185\). The superadiabatic surface layer is therefore treated as explained structure, not an unresolved anomaly. |
| mesa_model_0200_profile_05 | \(1.000\,M_\odot\) | 0.006 Gyr | 1.648 | -0.285 | Normal | Deviation is accounted for by the convective driver \(\Delta n_{conv}=0.375\). The superadiabatic surface layer is therefore treated as explained structure, not an unresolved anomaly. |
| mesa_model_0244_profile_06 | \(1.000\,M_\odot\) | 0.043 Gyr | 2.764 | 0.962 | Normal | The residual \(\delta_{global}=0.962\) remains below the screening threshold after subtracting the five physical deviation drivers. |
| mesa_model_0250_profile_07 | \(1.000\,M_\odot\) | 0.074 Gyr | 2.776 | 1.025 | Normal | The residual \(\delta_{global}=1.025\) remains below the screening threshold after subtracting the five physical deviation drivers. |
| mesa_model_0295_profile_08 | \(1.000\,M_\odot\) | 4.600 Gyr | 3.085 | 1.279 | Normal | The residual \(\delta_{global}=1.279\) remains below the screening threshold after subtracting the five physical deviation drivers. |

The current repository evidence contains the imported MESA-Web snapshots listed above. Additional 0.8, 2.0, and 5.0 \(M_\odot\) tracks are treated as the next multitrack expansion targets until their profile files are imported and recorded in the source manifest. This keeps the results section consistent with the actual evidence package while preserving the planned screening logic for a broader stellar array.

This screening demonstrates the purpose of the anomaly score: to find the needle in the haystack. Most stars can be structurally unusual during early evolution or near the surface, but they are still normal when the five physical drivers explain the deviation. A true candidate anomaly would appear only as a profile with a large \(|\delta_{global}|\), meaning that known thermodynamic and structural corrections are insufficient. Future work will apply the same pipeline to the proposed full MIST grid of approximately 50,000 stellar models to isolate the small fraction of stars, potentially around 1%, whose residuals suggest missing physics such as magnetic fields, rapid rotation, binary tides, or other effects not yet included in the present driver model.
