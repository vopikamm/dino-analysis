# Module containing the DINO experiment class collecting all diagnostics.
import xarray       as xr
import xgcm         as xg
import xnemogcm     as xn
import cftime as cft
import f90nml
import numpy        as np

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
        try:
            self.yearly         = self.open_data('DINO_1y_grid_*')
        except:
            self.yearly         = None
        try:
            self.monthly        = self.open_data('DINO_1m_grid_*')
        except:
            self.monthly        = None
        self.data           = xr.merge(
            [i for i in [self.yearly, self.monthly] if i is not None],
            compat='minimal'
        )
    
    def get_restarts(self):
        """" Detect restart folders if they exist and return their names."""
        restarts = []
        for paths in sorted(Path(self.path).iterdir()):
            if paths.is_dir():
                restarts.append(PurePath(paths).name)
        if len(restarts)==0:
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
    
    def open_namelist(self):
        """ Open the namelist_cfg as a f90nml dict."""
        namelist = f90nml.read(
            self.path + self.restarts[0] + '/namelist_cfg'
        )
        return(namelist)
    
    def open_restart(self, name):
        """ Open one or multiple restart files."""
        restart_files = []
        for paths in sorted(Path(self.path).iterdir()):
            if name in PurePath(paths).name:
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
        ds.assign_coords(dict(
            lon=(["y", "x"], self.domain.glamt.values),
            lat=(["y", "x"], self.domain.gphit.values)
        ))
        return(ds)

    def add_sigma_levels(self):
        """ Add coordinates for the sigma levels. """
        levels = np.mgrid[
            1:len(self.domain.z_f)+1,   # along vertical axis
            0:len(self.domain.y_c)  ,   # along j-axis
            0:len(self.domain.x_c)      # along i-axis
        ][0]
        self.domain['sigma_levels'] = (['z_f', 'y_c', 'x_c'], levels)
    
    def regrid_restart(self, from_restart, to_restart, to_netcdf=False):
        """Regridding a restart file to the horizontal resolution of another. """
        


    
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
        acc = (self.data.uoce.isel(x_f=0) * self.data.e3u.isel(x_f=0) * self.domain.e2u.isel(x_f=0)).sum(['y_c', 'z_c']) / 1e6
        return(acc)



