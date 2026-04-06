# Dataset: FFF Printer Full Lifecycle Vibration Monitoring and Tensile Test Results

This repository contains the experimental dataset used for long-term lifecycle monitoring of a fused filament fabrication (FFF) 3D printer. The dataset includes synchronized vibration measurements collected during printer operation and corresponding tensile test results of printed specimens over the printer’s service life.

## Repository Structure

### `Ender5_Vibration_Data/`
This folder contains vibration monitoring data collected from the Creality Ender 5 Plus printer during different stages of printer usage.

The data include:
- Raw vibration signals
- Multi-sensor acceleration measurements
- Monitoring samples collected across cumulative printing hours
- Time-series data files for condition tracking

These data are intended for:
- printer health monitoring
- wear progression analysis
- predictive maintenance research
- signal processing and machine learning studies

### `Tensile_test_results/`
This folder contains the tensile testing results of printed specimens corresponding to each monitored printer stage.

The data include:
- peak tensile force
- mechanical performance measurements
- repeated specimen test results
- statistical summaries for each monitoring sample

These results support the analysis of the relationship between printer wear and part mechanical performance degradation.

## Experimental Purpose

The dataset was developed to investigate the long-term degradation behavior of desktop FFF printers and its effect on printed part quality.

The primary objective is to support research in:
- in-situ monitoring
- printer wear assessment
- data-driven predictive maintenance
- quality prediction of additive manufacturing systems

## Printer Platform

- Printer: Creality Ender 5 Plus
- Process: Fused Filament Fabrication (FFF)
- Material: PLA
- Sensor Type: Multi-point accelerometers
- Test Method: Tensile testing of printed specimens
