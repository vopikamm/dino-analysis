# Module containing the DINO experiment class collecting all diagnostics.
import xarray       as xr
import xgcm         as xg
import xnemogcm     as xn
import cftime as cft
import f90nml
import numpy        as np
from xesmf import Regridder
from math import ceil, floor
import scipy.sparse as sparse
import scipy.sparse.linalg as la

from pathlib import Path, PurePath

class Experiment:
    """ Experiment class organizing data, transformations, diagnostics, etc... """
    def __init__(self, path, experiment_name, restarts=None):
        self.name           = experiment_name
        self.path           = path + experiment_name + '/'
        self.restarts       = self.get_restarts(restarts)
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
            self.ten_daily            = self.open_data('DINO_10d_grid_*')
        except:
            self.ten_daily            = None
        try:
            self.daily            = self.open_data('DINO_1d_grid_*')
        except:
            self.daily            = None
            #chunks.pop('t_0')
        self.data           = xr.merge(
            [i for i in [self.yearly, self.monthly, self.ten_daily, self.daily] if i is not None],
            compat='minimal'
        )#.chunk(chunks=chunks)
    
    def get_restarts(self, restarts):
        """" Detect restart folders if they exist and return their names."""
        if restarts is None:
            restarts = []
            try:
                for paths in sorted(Path(self.path).iterdir(), key=lambda x: float(x.name.strip('restart'))):
                    if paths.is_dir():
                        restarts.append(PurePath(paths).name)
            except:
                restarts.append('')
            return(restarts)
        else:
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

    def regrid_emp(self, other, name='emp_climatology.nc'):
        """
        Regridding the EmP-field (name) onto another grid (other).
        """
        # Create two EmP climatology datasets with the respective coordinates as self/other.
        emp_self = xr.open_dat

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
    
    def transform_to_density(self, var, isel={'t_y':-1}, z=2000):
        """Transforming a variable (vertical T-point: z_c) to density coordinates."""
        # Cut out bottom layer of z_c, such that z_f is outer (land anyway)
        ds_top = self.data.isel(z_c=slice(0,-1), **isel)

        # Compute density if necessary
        rho = self.get_rho(z=z).isel(z_c=slice(0,-1), **isel).rename('rho')
        # Mask boundary
        rho = rho.where(self.domain.tmask == 1.0)
        # define XGCM grid object with outer dimension z_f 
        grid = xg.Grid(ds_top,
            coords={
                "X": {"right": "x_f", "center":"x_c"},
                "Y": {"right": "y_f", "center":"y_c"},
                "Z": {"center": "z_c", "outer": "z_f"}
            },
            metrics=xn.get_metrics(ds_top),
            periodic=False
        )

        # Interpolate sigma2 on the cell faces
        rho_var = grid.interp_like(rho, var)
        rho_out = grid.interp(rho_var, 'Z',  boundary='extend')

        # Target values for density coordinate
        rho_tar = np.linspace(
            floor(rho_out.min().values),
            ceil(rho_out.max().values),
            36
        )
        # Transform variable to density coordinates:
        var_transformed = grid.transform(
            var,
            'Z',
            rho_tar,
            method='conservative',
            target_data=rho_out
        )
        return(var_transformed)

    def get_T_star(self):
        """ Compute the temperature restoring profile. """
        T_star = self.data.sst - (
            self.data.qns
            #+ self.data.empmr * self.data.sst * 3991.86795
            + self.data.qsr
        ) / self.namelist['namusr_def']['rn_trp']
        return(T_star.where(self.domain.tmask.isel(z_c=0) == 1.))
    
    def get_emp(self, d=20, tolerance=1e-20):
        """
        Compute the EmP-field (Evaporation - Precipitation).
        To ensure a conserved volume the global mean EmP should be zero.
        The mean is removed from the maximum precipitation in the tropics
        with a tapering such that the prescribed freshwater flux in
        high latitudes does not alter deep water formation.

        Tapering:  F(y) = A * (cos( pi * y / d ))    if |y| <= d
                        = 0                          else
    
        with        A   = emp_mean * L / (2 * d)
        """
        emp         = (self.data.saltflx / self.data.sss)
        
        emp_mean    = self.grid.average(emp, ['X', 'Y'])
        taper       = ( 
            (abs(self.domain.gphit) <= 20)                  # one in tropics, zero elsewhere
            * (np.cos( np.pi * self.domain.gphit / d) + 1)  # tapering-function
        )
        taper_mean  = self.grid.average(taper, ['X', 'Y'])
        while abs(emp_mean).max().values > tolerance:  
            emp_star = emp - emp_mean * taper / taper_mean
            emp      = emp_star
            emp_mean = self.grid.average(emp_star, ['X', 'Y'])
        return(emp_star)

    def get_S_star(self):
        """ Compute the salinity restoring profile. """
        if 'saltflx' in list(self.data.keys()):
            S_star = self.data.sss - self.data.saltflx  / self.namelist['namusr_def']['rn_srp']
            return(S_star.where(self.domain.tmask.isel(z_c=0) == 1.))
        else:
            print('Warning: saltflux is not in the dataset. Assumed shape by Romain Caneill (2022).')
            return(37.12 * np.exp(- self.domain.gphit**2 / 260.**2 ) - 1.1 * np.exp( - self.domain.gphit**2 / 7.5**2 ))
    
    def get_rho_star(self):
        """
        Compute the density restoring profile from salinity and temperature restoring.
        Referenced to surface pressure.
        """
        T_star = self.get_T_star()
        S_star = self.get_S_star()
        nml = self.namelist['nameos']
        if nml['ln_seos']:
            rho_star = (
                - nml['rn_a0'] * (1. + 0.5 * nml['rn_lambda1'] * ( T_star - 10.)) * ( T_star - 10.)
                + nml['rn_b0'] * (1. - 0.5 * nml['rn_lambda2'] * ( S_star - 35.)) * ( S_star - 35.)
                - nml['rn_nu'] * ( T_star - 10.) * ( S_star - 35.)
            ) + 1026
            return(rho_star)
        else:
            raise Exception('Only S-EOS has been implemented yet.')
    
    def get_rho(self, z=0.):
        """
        Compute potential density referencedto the surface according to the EOS. 
        Uses gdepth_0 as depth for simplicity and allows only for S-EOS currently.

            z = 0.          Reference pressure level [m]
        """
        nml  = self.namelist['nameos']
        # masking of T,S
        soce = self.data.soce.where(self.domain.tmask == 1.)
        toce = self.data.toce.where(self.domain.tmask == 1.)
        if nml['ln_seos']:
            rho = (
                - nml['rn_a0'] * (1. + 0.5 * nml['rn_lambda1'] * ( toce - 10.) + nml['rn_mu1'] * z) * ( toce - 10.) 
                + nml['rn_b0'] * (1. - 0.5 * nml['rn_lambda2'] * ( soce - 35.) - nml['rn_mu2'] * z) * ( soce - 35.) 
                - nml['rn_nu'] * ( toce - 10.) * ( soce - 35.)
            ) + 1026
            return(rho)
        else:
            raise Exception('Only S-EOS has been implemented yet.')
        
    def get_N_squared(self):
        """
        Compute the squared Brunt-Väisälä frequency according to the EOS. 
        Only for S-EOS currently.
        """
        nml  = self.namelist['nameos']
        # masking of T, S, z
        soce = self.data.soce.where(self.domain.tmask == 1.)
        toce = self.data.toce.where(self.domain.tmask == 1.)
        z    = self.domain.gdept_0.where(self.domain.tmask == 1.)

        if nml['ln_seos']:
            alpha   = ( nml['rn_a0'] * (1. + nml['rn_lambda1'] * ( toce - 10.) + nml['rn_mu1'] * z) + nml['rn_nu'] * soce ) / 1026.  
            beta    = ( nml['rn_b0'] * (1. - nml['rn_lambda2'] * ( soce - 35.) - nml['rn_mu2'] * z) + nml['rn_nu'] * toce ) / 1026.
            Nsq   = 9.80665 * (
                - self.grid.interp(alpha, 'Z', boundary='extend')           # alpha on W
                * self.grid.diff(toce, 'Z', boundary='extend')              # dT/dz
                + self.grid.interp(beta, 'Z', boundary='extend')            # beta on W
                * self.grid.diff(soce, 'Z', boundary='extend')              # dS/dz
            ) / self.data.e3w
            Nsq = Nsq.where(Nsq >= 1e-8).fillna(1e-7)
            return(Nsq)
        else:
            raise Exception('Only S-EOS has been implemented yet.')
    
    def get_BTS(self):
        """ Compute the BaroTropic Streamfunction. """
        # Interpolating u, e3u on f-points
        u_on_f  = self.grid.interp(self.data.uoce, 'Y')
        e3_on_f = self.grid.interp(self.data.e3u, 'Y')
        # Vertical integral
        U = (u_on_f * e3_on_f).sum('z_c')
        # Cumulative integral over y
        bts = (U[:,::-1,:] * self.domain.e2f[::-1,:]).cumsum('y_f') / 1e6
        return(bts)
    
    def get_MOC(self, var, isel={'t_y':-1}, z=2000):
        """ Compute the Meridional Overturning Streamfunction of transport variable `var`. """
        # Prepare the meridional transport:
        if var.name == 'vocetr_eff':
            var = var.isel(z_c=slice(0,-1), **isel)
        else:
            var = (var * self.data.e3v * self.domain.e1v).isel(z_c=slice(0,-1), **isel)
        var_tra = self.transform_to_density(var=var, isel=isel, z=z)
        moc = var_tra.sum(dim='x_c')[...,::-1].cumsum('rho') / 1e6
        moc = moc.assign_coords(dict({'y_f': self.domain.gphif.isel(x_f=0).values}))
        return(moc)
    
    def get_ACC(self):
        """
        Compute the Antarctic Circumpolar Current.

        Defined as the volume transport through the zonal boundaries.        
        """
        acc = (self.data.uoce.isel(x_f=0) * self.domain.e3u_0.isel(x_f=0) * self.domain.e2u.isel(x_f=0)).sum(['y_c', 'z_c']) / 1e6
        return(acc)
    
    @staticmethod
    def _get_dynmodes(Nsq, e3t, e3w, nmodes=2):
        """
        Calculate the 1st nmodes ocean dynamic vertical modes.
        Based on
        http://woodshole.er.usgs.gov/operations/sea-mat/klinck-html/dynmodes.html
        by John Klinck, 1999.
        """
        # 2nd derivative matrix plus boundary conditions
        Ndz     = (Nsq * e3w)
        e3t     = e3t
        #Ndz_m1  = np.roll(Ndz, -1)
        #e3t_p1  = np.roll(e3t, 1)
        d0  = np.r_[1. / Ndz[1] / e3t[0],
                   (1. / Ndz[2:-1] + 1. / Ndz[1:-2]) / e3t[1:-2],
                   1. / Ndz[-2] / e3t[-2]]
        d1  = np.r_[0., -1. / Ndz[1:-1] / e3t[1:-1]]
        dm1 = np.r_[-1. / Ndz[1:-1] / e3t[0:-2], 0.]
        diags = 1e-4 * np.vstack((d0, d1, dm1))
        d2dz2 = sparse.dia_matrix((diags, (0, 1, -1)), shape=(len(Nsq)-1, len(Nsq)-1))
        # Solve generalized eigenvalue problem for eigenvalues and vertical
        # Horizontal velocity modes
        try:
            eigenvalues, modes = la.eigs(d2dz2, k=nmodes+1, which='SM')
            mask = (eigenvalues.imag == 0) & (eigenvalues >= 1e-10)
            eigenvalues = eigenvalues[mask]
            # Sort eigenvalues and modes and truncate to number of modes requests
            index = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[index[:nmodes]].real
            # Modal speeds
            ce = 1 / np.sqrt(eigenvalues * 1e4)
        except:
            ce = -np.ones(nmodes)

        return(ce)

    def get_vmodes(self, nmodes=2):
        """ compute vertical modes
        Wrapper for calling `compute_vmodes` with DataArrays through apply_ufunc. 
        z levels must be in descending order (first element is at surface, last element is at bottom) with algebraic depth (i.e. negative)
        Normalization is performed here (int_z \phi^2 \dz = Hbot)

        Parameters:
        ___________
        nmodes: int, optional
            number of vertical baroclinic modes (barotropic is added)

        Returns:
        ________
        xarray.DataSet: vertical modes (p and w) and eigenvalues
        !! (currently only eigenvalues)
        _________
        """
        Nsq = (self.get_N_squared())
        res = xr.apply_ufunc(self._get_dynmodes, 
                             Nsq.isel(x_c=slice(1,-1), y_c=slice(1,-1), t_y=-1).chunk({'z_f':-1}),
                             self.data.e3t.where(self.domain.tmask==1.0).isel(x_c=slice(1,-1), y_c=slice(1,-1), t_y=-1).chunk({'z_c':-1}),
                             self.data.e3w.isel(x_c=slice(1,-1), y_c=slice(1,-1), t_y=-1).chunk({'z_f':-1}),
                             input_core_dims=[['z_f'],['z_c'],['z_f']],
                             dask='parallelized', vectorize=True,
                             output_dtypes=[Nsq.dtype],
                             output_core_dims=[["mode"]],
                             dask_gufunc_kwargs={"output_sizes":{"mode":nmodes}}
                            )
        return res
    
    def get_Zanna_Bolton(self, gamma=1.0):
        """
        Implementation of the Zanna & Bolton (2020) subgrid closure discovered by a machine learning algorithm.
        The discretization of its operators follows Perezhogin.
        """
        dudx        = self.grid.diff(self.data.uoce * self.domain.umask / self.domain.e2u, 'X') * self.domain.e2t / self.domain.e1t
        dvdy        = self.grid.diff(self.data.voce * self.domain.vmask / self.domain.e1v, 'Y') * self.domain.e1t / self.domain.e2t

        dudy        = self.grid.diff(self.data.uoce / self.domain.e1u, 'Y') * self.domain.e1f / self.domain.e2f * self.domain.fmask
        dvdx        = self.grid.diff(self.data.voce / self.domain.e2v, 'X') * self.domain.e1f / self.domain.e1f * self.domain.fmask

        sh_xx       = dudx - dvdy       # Stretching deformation \tilde{D} on T-point
        sh_xy       = dvdx + dudy       # Shearing deformation D on F-point 
        vort_xy     = dvdx - dudy       # Relative vorticity \Zeta on F-point

        kappa_t     = self.domain.e2t * self.domain.e1t * self.domain.tmask * gamma
        kappa_f     = self.domain.e2f * self.domain.e1f * self.domain.fmask * gamma

        # Interpolating defomation and vorticity on opposite grid-points
        # TODO: different discretizations of the interpolation as proposed by Pavel
        vort_xy_t   = self.grid.interp(vort_xy,['X', 'Y']) * self.domain.tmask
        sh_xy_t     = self.grid.interp(sh_xy,['X', 'Y']) * self.domain.tmask
        sh_xx_f     = self.grid.interp(sh_xx,['X', 'Y']) * self.domain.fmask

        # Hydrostatic component of Txx/Tyy
        sum_sq      = 0.5 * (vort_xy_t**2 + sh_xy_t**2 + sh_xx**2)
        # Deviatoric component of Txx/Tyy        
        vort_sh     = vort_xy_t * sh_xy_t

        Txx         = - kappa_t * (- vort_sh + sum_sq)
        Tyy         = - kappa_t * (+ vort_sh + sum_sq)
        Txy         = - kappa_f * (vort_xy * sh_xx_f)

        ZB2020u     = (self.grid.diff(Txx * self.data.e3t * self.domain.e2t**2, 'X') / self.domain.e2u      \
                + self.grid.diff(Txy * self.domain.e3f * self.domain.e1f**2, 'Y') / self.domain.e1u)        \
                / (self.domain.e1u * self.domain.e2u) / (self.data.e3u + 1e-70)

        ZB2020v     = (self.grid.diff(Txy * self.data.e3f * self.domain.e2f**2, 'X') / self.domain.e2v      \
                + self.grid.diff(Tyy * self.data.e3t * self.domain.e1t**2, 'Y') / self.domain.e1v)          \
                / (self.domain.e1v * self.domain.e2v) / (self.data.e3v+1e-70)
        
        return {
            'ZB2020u': ZB2020u, 'ZB2020v': ZB2020v, 
            'Txx': Txx, 'Tyy': Tyy, 'Txy': Txy, 
            'sh_xx': sh_xx, 'sh_xy': sh_xy, 'vort_xy': vort_xy,
        }


