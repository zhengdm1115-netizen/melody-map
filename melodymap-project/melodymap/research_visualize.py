from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches='tight'); plt.close(fig)


def plot_case_map(venues, sensors, distances, path: Path, radius_m=500):
    fig, ax = plt.subplots(figsize=(8,7))
    ax.scatter(venues.longitude, venues.latitude, s=70, label='Live-music venues')
    ax.scatter(sensors.longitude, sensors.latitude, s=90, marker='x', label='Pedestrian sensors')
    for _, r in distances[distances.within_radius].iterrows():
        v = venues[venues.venue_name==r.venue_name].iloc[0]
        s = sensors[sensors.sensor_description==r.sensor_description].iloc[0]
        ax.plot([v.longitude,s.longitude],[v.latitude,s.latitude], alpha=.25, linewidth=1)
    for _, r in venues.iterrows(): ax.text(r.longitude, r.latitude, r.venue_name, fontsize=8, ha='left', va='bottom')
    for _, r in sensors.iterrows(): ax.text(r.longitude, r.latitude, r.sensor_description, fontsize=7, ha='right', va='top')
    ax.set_title(f'Real case study: venue–sensor links within {radius_m} m')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude'); ax.legend(fontsize=8)
    _save(fig,path)


def plot_distance_matrix(distances, path: Path):
    p = distances.pivot(index='venue_name', columns='sensor_description', values='distance_m')
    fig, ax = plt.subplots(figsize=(9,4.8))
    im=ax.imshow(p.values, aspect='auto')
    ax.set_xticks(range(len(p.columns)), [c.replace(' ','\n',1) for c in p.columns], fontsize=8)
    ax.set_yticks(range(len(p.index)), p.index)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]): ax.text(j,i,f'{p.iloc[i,j]:.0f} m',ha='center',va='center',fontsize=8)
    ax.set_title('Distance from each music venue to selected pedestrian sensors')
    fig.colorbar(im, ax=ax, label='Distance (m)')
    _save(fig,path)


def plot_sensor_reference(stats, path: Path):
    x=stats.sort_values('mean_hourly_pedestrians')
    fig,ax=plt.subplots(figsize=(8,4.8)); ax.barh(x.sensor_description,x.mean_hourly_pedestrians)
    ax.set_xlabel('Published long-run mean hourly pedestrian count')
    ax.set_title('Real pedestrian-volume reference for selected sensors')
    _save(fig,path)


def plot_simulation_panel(summary: pd.DataFrame, experiment: str, path: Path, xlabel: str):
    x=summary[summary.experiment==experiment].sort_values('level')
    fig,ax=plt.subplots(figsize=(7.5,4.8))
    ax.plot(x.level,x.mean_rho,marker='o')
    ax.fill_between(x.level, x.q10_rho, x.q90_rho, alpha=.18, label='10–90% across simulations')
    ax.axhline(0,linewidth=1,alpha=.5)
    ax.set_xlabel(xlabel); ax.set_ylabel('Recovered Spearman association')
    titles={
        'coverage':'Sensitivity to pedestrian-sensor coverage',
        'radius':'Sensitivity to spatial analysis radius',
        'coordinate_noise':'Sensitivity to coordinate noise',
        'confounding':'Spurious association from shared urban centrality',
        'effect_size':'Recovery as the known music effect increases',
    }
    ax.set_title(titles.get(experiment,experiment)); ax.legend(fontsize=8)
    _save(fig,path)
