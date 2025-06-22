# MXFP Fake Quantisation for QAT (Only FP4 and FP6..)

## Format names and parameters of concrete MX-compliant formats.
| Format Name  | Element Data Type | Element Bits (d) | Scaling Block Size (k) | Scale Data Type | Scale Bits (w) |
|---------|-------------------|------------------|-------------------------|------------------|----------------|
| MXFP6   |  FP6 (E3M2)         | 6                | 32                      | E8M0             | 8              |
|               | FP6 (E2M3)         |                  |                         |                  |                |
| MXFP4   | FP4 (E2M1)         | 4                | 32                      | E8M0             | 8              |


## MXFP4 Encoding Details
| Description     | Representation           |
|-----------------|--------------------------|
| Exponent bias   | 1                        |
| Infinities      | N/A                      |
| NaN             | N/A                      |
| Zeros           | S 00 02                  |
| Max normal      | S 11 12 = ± 2² × 1.5 = ± 6.0 |
| Min normal      | S 01 02 = ± 2⁰ × 1.0 = ± 1.0 |
| Max subnorm     | S 00 12 = ± 2⁰ × 0.5 = ± 0.5 |
| Min subnorm     | S 00 12 = ± 2⁰ × 0.5 = ± 0.5 |


## MXFP6 Encoding Details
| Description     | E2M3                             | E3M2                                |
|-----------------|----------------------------------|-------------------------------------|
| Exponent bias   | 1                                | 3                                   |
| Infinities      | N/A                              | N/A                                 |
| NaN             | N/A                              | N/A                                 |
| Zeros           | S 00 000₂                        | S 000 00₂                           |
| Max normal      | S 11 111₂ = ± 2² × 1.875 = ± 7.5 | S 111 11₂ = ± 2⁴ × 1.75 = ± 28.0    |
| Min normal      | S 01 000₂ = ± 2⁰ × 1.0 = ± 1.0   | S 001 00₂ = ± 2⁻² × 1.0 = ± 0.25    |
| Max subnorm     | S 00 111₂ = ± 2⁰ × 0.875 = ± 0.875 | S 000 11₂ = ± 2⁻² × 0.75 = ± 0.1875 |
| Min subnorm     | S 00 001₂ = ± 2⁰ × 0.125 = ± 0.125 | S 000 01₂ = ± 2⁻² × 0.25 = ± 0.0625 |


## Imprtant Calculations
### OCP Bias
$$2^{(\text{exponent\_bits} - 1)} - 1$$

### E_MAX
$$2^{\text{exponent\_bits}} - 1 - \text{bias}$$

### E8M0 Scale 
$$S = 2^s$$
Where $s = \lfloor \log_2 M \rfloor - e\_max$; and $M = max_{1 \leq i \leq 32} |X_i|$ 

## Error Observations
For this purpose a weight matrix of size ~ 2.35M was used (1536, 1536)

| MXFP Format | MSE | SNR (dB) |
|-------------|-----|----------|
| MXFP4 (E2M1) | 4.05e-05 | 17.625 |
| MXFP6 (E2M3) | **2.08e-06** | **30.625** |
| MXFP6 (E3M2) | 6.70e-06 | 25.5 |

## References
1. https://arxiv.org/pdf/2310.10537
2. https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
3. https://arxiv.org/html/2502.20853v1

