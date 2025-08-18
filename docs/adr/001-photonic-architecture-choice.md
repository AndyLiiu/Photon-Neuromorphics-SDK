# ADR-001: Silicon Photonic Architecture Choice

Date: 2025-01-15

## Status

Accepted

## Context

We needed to choose a photonic computing architecture for neural network implementations. The main options were:

1. **Silicon Photonics**: CMOS-compatible, mature fabrication, wavelength ~1550nm
2. **Lithium Niobate**: High-speed modulation, but expensive fabrication
3. **Plasmonic**: Ultra-compact, but high losses
4. **Free-space Optics**: Parallel processing, but alignment sensitive

The decision impacts:
- Hardware compatibility and cost
- Performance characteristics  
- Manufacturing scalability
- Integration with existing infrastructure

## Decision

We chose **Silicon Photonics** as the primary architecture with the following rationale:

1. **CMOS Compatibility**: Leverages existing semiconductor infrastructure
2. **Mature Ecosystem**: Established foundries (AIM Photonics, IMEC, etc.)
3. **Wavelength Selection**: 1550nm telecom band with low-loss fibers
4. **Component Library**: Rich set of validated building blocks (MZI, microrings, etc.)
5. **Integration Path**: Clear path to electronic-photonic co-design

## Consequences

### Positive Consequences

- **Lower Manufacturing Cost**: CMOS fab compatibility reduces production costs
- **Proven Components**: Well-characterized building blocks accelerate development
- **Industry Standards**: Aligns with telecom and datacom infrastructure
- **Scalability**: Clear path to wafer-scale integration
- **Ecosystem Support**: Extensive tool support and foundry services

### Negative Consequences

- **Speed Limitations**: Thermal tuning limits switching speeds to ~MHz
- **Power Requirements**: Thermal phase shifters consume significant power
- **Temperature Sensitivity**: Requires active thermal management
- **Loss Budget**: Insertion losses limit network depth without amplification

## Alternatives Considered

- **Lithium Niobate**: Rejected due to high fabrication cost and limited CMOS compatibility
- **Plasmonic**: Rejected due to excessive propagation losses
- **Free-space Optics**: Rejected due to alignment sensitivity and packaging complexity

## Implementation Notes

- Target AIM Photonics Multi-Project Wafer (MPW) runs for prototyping
- Design for 220nm SOI process with standard component library
- Plan for hybrid integration with electronics for high-speed control
- Consider electro-optic materials for future speed improvements

## References

- [AIM Photonics Process Design Kit](https://www.aimphotonics.com/pdk)
- Shen et al., "Deep learning with coherent nanophotonic circuits", Nature Photonics (2017)
- [Silicon Photonics Design Guidelines](https://optics.org/silicon-photonics)