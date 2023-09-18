# Module containing the DINO experiment class collecting all diagnostics.
import xarray       as xr
import xgcm         as xg
import xnemogcm     as xn
import cftime as cft
import f90nml
import numpy        as np
from xesmf import Regridder

from pathlib import Path, PurePath

class Experiment:
    """ Experiment class organizing data, transformations, diagnostics, etc... """
    def __init__(self, path, experiment_name):
        self.name           = experiment_name
        self.path           = path + experiment_name + '/'
        self.restarts       = self.get_restarts()
        self.domain         = self.open_domain()
        self.namelist       = self.open_namelist()
        self.grid           = xg.Grid(self.domain, metrics=xn.get_metrics(self.domain), periodic=False)
        #chunks              = dict({'t_y':10, 't_m':120, 't_0':360})
        try:
            self.yearly         = self.open_data('DINO_1y_grid_*')
        except:
            self.yearly         = None
            #chunks.pop('t_y')
        try:
            self.monthly        = self.open_data('DINO_1m_grid_*')
        except:
            self.monthly        = None
            #chunks.pop('t_m')
        try:
            self.d10            = self.open_data('DINO_10d_grid_*')
        except:
            self.d10            = None
            #chunks.pop('t_0')
        self.data           = xr.merge(
            [i for i in [self.yearly, self.monthly, self.d10] if i is not None],
            compat='minimal'
        )#.chunk(chunks=chunks)
    
    def get_restarts(self):
        """" Detect restart folders if they exist and return their names."""
        restarts = []
        try:
            for paths in sorted(Path(self.path).iterdir(), key=lambda x: float(x.name.strip('restart'))):
                if paths.is_dir():
                    restarts.append(PurePath(paths).name)
        except:
            restarts.append('')
        return(restarts)
    
    def open_domain(self):
        """ Open the domain_cfg not lazy."""
        domain = xn.open_domain_cfg(
            datadir=Path(self.path + self.restarts[0])
        )
        return(domain)

    def open_data(self, file_name):
        """ Open the data lazy. """
        Data = []
        for restart in self.restarts:
            files     = Path(self.path + restart).glob(file_name)
            Data.append( xn.open_nemo(domcfg=self.domain, files=files).chunk())
        if Data:
            return(xr.concat(Data,  "t").rename({'t': 't_' + file_name[6] }))
        else:
            return(None)
    
    def open_namelist(self, restart=0):
        """ Open the namelist_cfg as a f90nml dict."""
        namelist = f90nml.read(
            self.path + self.restarts[restart] + '/namelist_cfg'
        )
        return(namelist)
    
    def open_restart(self, restart_path=None):
        """ Open one or multiple restart files."""
        restart_files = []
        for paths in sorted(Path(self.path).iterdir()):
            if (str(self.namelist['namrun']['nn_itend']) + '_restart') in PurePath(paths).name:
                restart_files.append(PurePath(paths))

        ds = xr.open_mfdataset(
            restart_files,
            preprocess=xn.domcfg.domcfg_preprocess,
            combine_attrs="drop_conflicts",
            data_vars="minimal",
            drop_variables=["x", "y"],
        )
        for i in [
            "DOMAIN_position_first",
            "DOMAIN_position_last",
            "DOMAIN_number",
            "DOMAIN_number_total",
            "DOMAIN_size_local",
        ]:
            ds.attrs.pop(i, None)
        ds = ds.assign_coords(dict(
            lon=(["y", "x"], self.domain.glamt.values),
            lat=(["y", "x"], self.domain.gphit.values)
        )).drop_vars(['x', 'y'])
        return(ds)

    def add_sigma_levels(self):
        """ Add coordinates for the sigma levels. """
        levels = np.mgrid[
            1:len(self.domain.z_f)+1,   # along vertical axis
            0:len(self.domain.y_c)  ,   # along j-axis
            0:len(self.domain.x_c)      # along i-axis
        ][0]
        self.domain['sigma_levels'] = (['z_f', 'y_c', 'x_c'], levels)
    
    def regrid_restart(self, other):
        """Regridding a restart file to the horizontal resolution of another. """
        _lr = self.open_restart()
        lr = self.extrapolate_restart_on_land(lr=_lr)
        hr = other.open_restart()

        dt = other.namelist['namdom']['rn_Dt']

        # initiate Regridder
        regridder = Regridder(lr, hr, "bilinear",  extrap_method="nearest_s2d", ignore_degenerate=True)
        restart_regrid = regridder(lr)

        # apply high resolution mask
        restart_regrid *= xr.where(hr.tn.isel(nav_lev=0, time_counter=0)==0.0, 0, 1)

        # set velocities to zero
        restart_regrid['ub'].loc[:] = 0.0
        restart_regrid['un'].loc[:] = 0.0
        restart_regrid['vb'].loc[:] = 0.0
        restart_regrid['vn'].loc[:] = 0.0

        # assign missing values from hr/lr dataset and change time_step
        restart_regrid['lon'] = hr.lon
        restart_regrid['lat'] = hr.lat
        restart_regrid['kt'] = _lr.kt
        restart_regrid['ndastp'] = _lr.ndastp
        restart_regrid['adatrj'] = _lr.adatrj
        restart_regrid['ntime'] = _lr.ntime
        restart_regrid['rdt'] = dt

        # order as the high resolution datset
        restart_regrid = restart_regrid[list(hr.keys())]

        return(restart_regrid)

    def extrapolate_restart_on_land(self, lr):
        """ 
        Extrapolating a restart dataset onto land points.
        This is necessary for LR --> HR regridding.
        """
        res = self.namelist['namusr_def']['rn_e1_deg']
        # Create a dummy datasat on the land-points
        temp_p  = np.concatenate((lr.tn.values, lr.tn.values[:,:,:,-2:]+2*res), axis=3)
        lon_p   = np.concatenate((lr.lon.values, lr.lon.values[:,-2:]+2*res), axis=1)
        lat_p   = np.concatenate((lr.lat.values, lr.lat.values[:,-2:]+2*res), axis=1)
        # define data with variable attributes
        data_vars = {
            'temperature':(
                ['time_counter', 'nav_lev', 'y_c', 'x_c'], temp_p, 
                                 {'units': 'C'}
            )
        }
        # define coordinates
        coords = {  'time_counter': ('time_counter', lr.time_counter.values),
                    'nav_lev': ('nav_lev', lr.nav_lev.values),
                    'lat': (['y', 'x'], lat_p),
                    'lon': (['y', 'x'], lon_p)
                  }
        # create dataset
        ds_lr = xr.Dataset(data_vars=data_vars, 
                        coords=coords, 
        )
        # Initiate extrapolator
        extrapolator = Regridder(lr.isel(x=slice(1,-1), y=slice(1,-1)), ds_lr, "nearest_s2d")

        return(extrapolator(lr.isel(x=slice(1,-1), y=slice(1,-1))))
    
    def get_T_star(self):
        """ Compute the temperature restoring profile. """
        return((self.data.qns + self.data.empmr * self.data.sst * 3991.86795 + self.data.qsr) / ( 40.) + self.data.sst)
    
    def get_rho(self):
        """
        Compute potential density referencedto the surface according to the EOS. 
        Uses gdepth_0 as depth for simplicity and allows only for S-EOS currently.
        """
        nml = self.namelist['nameos']
        if nml['ln_seos']:
            rho = (
                - nml['rn_a0'] * (1. + 0.5 * nml['rn_lambda1'] * ( self.data.toce - 10.) + nml['rn_mu1'] * self.domain.gdept_0) * ( self.data.toce - 10.)
                + nml['rn_b0'] * (1. - 0.5 * nml['rn_lambda2'] * ( self.data.soce - 35.) - nml['rn_mu2'] * self.domain.gdept_0) * ( self.data.soce - 35.)
                - nml['rn_nu'] * ( self.data.toce - 10.) * ( self.data.soce - 35.)
            ) + 1026
            return(rho.where(rho > 0))
        else:
            raise Exception('Only S-EOS has been implemented yet.')
    
    def get_BTS(self):
        """ Compute the BaroTropic Streamfunction. """
        # Copy data
        _data_grid          = self.domain.copy()
        # Adding velocities and e3s to the data_grid
        _data_grid['uoce']  = (['t_y','z_c', 'y_c', 'x_f'], self.data.uoce.data)
        _data_grid['e3u']   = (['t_y','z_c', 'y_c', 'x_f'], self.data.e3u.data)
        # Interpolating on u_f
        _data_grid['u_on_f'] = self.grid.interp(_data_grid.uoce, 'Y')
        # Integrating over depth
        _data_grid['U']      = self.grid.integrate(_data_grid.u_on_f, 'Z')
        # Cumulative integral over x
        return((_data_grid.U[-1,::-1,:] * _data_grid.e2f[::-1,:]).cumsum('y_f') / 1e6)
    
    def get_ACC(self):
        """
        Compute the Antarctic Circumpolar Current.

        Defined as the volume transport through the zonal boundaries.        
        """
        acc = (self.data.uoce.isel(x_f=0) * self.domain.e3u_0.isel(x_f=0) * self.domain.e2u.isel(x_f=0)).sum(['y_c', 'z_c']) / 1e6
        return(acc)



