from covid_constants_and_util import *
from scipy.stats import pearsonr
#import fiona
#import geopandas
import os
#from geopandas.tools import sjoin
import time
import pandas as pd
import numpy as np


class CensusBlockGroups:
    """
    A class for loading geographic and demographic data from the ACS.

    A census block group is a relatively small area.
    Less good than houses but still pretty granular. https://en.wikipedia.org/wiki/Census_block_group

    Data was downloaded from https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-data.html
    We use the most recent ACS 5-year estimates: 2013-2017, eg:
    wget https://www2.census.gov/geo/tiger/TIGER_DP/2017ACS/ACS_2017_5YR_BG.gdb.zip
    These files are convenient because they combine both geographic boundaries + demographic data, leading to a cleaner join.

    The main method for data access is get_demographic_stats_of_point. Sample usage:
    x = CensusBlockGroups(gdb_files=['ACS_2017_5YR_BG_51_VIRGINIA.gdb'])
    x.get_demographic_stats_of_points(latitudes=[38.8816], longitudes=[-77.0910], desired_cols=['p_black', 'p_white', 'mean_household_income'])
    """
    def __init__(
        self, 
        base_directory=PATH_TO_CENSUS_BLOCK_GROUP_DATA,
        gdb_files=None,
        county_to_msa_mapping_filepath=PATH_TO_COUNTY_TO_MSA_MAPPING):
        self.base_directory = base_directory
        if gdb_files is None:
            self.gdb_files = ['ACS_2017_5YR_BG.gdb']
        else:
            self.gdb_files = gdb_files
        self.crs_to_use = WGS_84_CRS # https://epsg.io/4326, WGS84 - World Geodetic System 1984, used in GPS.
        self.county_to_msa_mapping_filepath = county_to_msa_mapping_filepath
        
        # NEW_V2
        self.load_raw_dataframes() # Load in raw geometry and demographic dataframes.

        # annotate demographic data with more useful columns.
        self.annotate_with_race()
        self.annotate_with_income()
        self.annotate_with_counties_to_msa_mapping()
        self.annotate_with_area_and_pop_density()

    def annotate_with_area_and_pop_density(self):
        # https://gis.stackexchange.com/questions/218450/getting-polygon-areas-using-geopandas. 
        # See comments about using cea projection. 
        gdf = self.geometry_d[['geometry']].copy().to_crs({'proj':'cea'})
        area_in_square_meters = gdf['geometry'].area.values
        self.block_group_d['block_group_area_in_square_miles'] = area_in_square_meters / (1609.34 ** 2)
        self.block_group_d['people_per_mile'] = (self.block_group_d['B03002e1'] /
                                               self.block_group_d['block_group_area_in_square_miles'])
        print(self.block_group_d[['block_group_area_in_square_miles', 'people_per_mile']].describe())


    def annotate_with_race(self):
        """
        Analysis focuses on black and non-white population groups. Also annotate with p_asian because of possible anti-Asian discrimination. 
        B03002e1  HISPANIC OR LATINO ORIGIN BY RACE: Total: Total population -- (Estimate)
        B03002e3  HISPANIC OR LATINO ORIGIN BY RACE: Not Hispanic or Latino: White alone: Total population -- (Estimate)
        B03002e4  HISPANIC OR LATINO ORIGIN BY RACE: Not Hispanic or Latino: Black or African American alone: Total population -- (Estimate)
        B03002e6  HISPANIC OR LATINO ORIGIN BY RACE: Not Hispanic or Latino: Asian alone: Total population -- (Estimate)
        """
        print("annotating with race")
        self.block_group_d['p_black'] = self.block_group_d['B03002e4'] / self.block_group_d['B03002e1']
        self.block_group_d['p_white'] = self.block_group_d['B03002e3'] / self.block_group_d['B03002e1']
        self.block_group_d['p_asian'] = self.block_group_d['B03002e6'] / self.block_group_d['B03002e1']
        print(self.block_group_d[['p_black', 'p_white', 'p_asian']].describe())

    def load_raw_dataframes(self):
        """
        Read in the original demographic + geographic data.
        """
        self.block_group_d = None
        self.geometry_d = None
        demographic_layer_names = ['X25_HOUSING_CHARACTERISTICS', 'X01_AGE_AND_SEX', 'X03_HISPANIC_OR_LATINO_ORIGIN', 'X19_INCOME']
        for file in self.gdb_files:
            # https://www.reddit.com/r/gis/comments/775imb/accessing_a_gdb_without_esri_arcgis/doj9zza
            full_path = os.path.join(self.base_directory, file)
            layer_list = fiona.listlayers(full_path)
            print(file)
            print(layer_list)
            geographic_layer_name = [a for a in layer_list if a[:15] == 'ACS_2017_5YR_BG']
            assert len(geographic_layer_name) == 1
            geographic_layer_name = geographic_layer_name[0]

            geographic_data = geopandas.read_file(full_path, layer=geographic_layer_name).to_crs(self.crs_to_use)
            # by default when you use the read file command, the column containing spatial objects is named "geometry", and will be set as the active column.
            print(geographic_data.columns)
            geographic_data = geographic_data.sort_values(by='GEOID_Data')[['GEOID_Data', 'geometry', 'STATEFP', 'COUNTYFP', 'TRACTCE']]
            for demographic_idx, demographic_layer_name in enumerate(demographic_layer_names):
                assert demographic_layer_name in layer_list
                if demographic_idx == 0:
                    demographic_data = geopandas.read_file(full_path, layer=demographic_layer_name)
                else:
                    old_len = len(demographic_data)
                    new_df = geopandas.read_file(full_path, layer=demographic_layer_name)
                    assert sorted(new_df['GEOID']) == sorted(demographic_data['GEOID'])
                    demographic_data = demographic_data.merge(new_df, on='GEOID', how='inner')
                    assert old_len == len(demographic_data)
            demographic_data = demographic_data.sort_values(by='GEOID')

            shared_geoids = set(demographic_data['GEOID'].values).intersection(set(geographic_data['GEOID_Data'].values))
            print("Length of demographic data: %i; geographic data %i; %i GEOIDs in both" % (len(demographic_data), len(geographic_data), len(shared_geoids)))

            demographic_data = demographic_data.loc[demographic_data['GEOID'].map(lambda x:x in shared_geoids)]
            geographic_data = geographic_data.loc[geographic_data['GEOID_Data'].map(lambda x:x in shared_geoids)]

            demographic_data.index = range(len(demographic_data))
            geographic_data.index = range(len(geographic_data))

            assert (geographic_data['GEOID_Data'] == demographic_data['GEOID']).all()
            assert len(geographic_data) == len(set(geographic_data['GEOID_Data']))


            if self.block_group_d is None:
                self.block_group_d = demographic_data
            else:
                self.block_group_d = pd.concat([self.block_group_d, demographic_data])

            if self.geometry_d is None:
                self.geometry_d = geographic_data
            else:
                self.geometry_d = pd.concat([self.geometry_d, geographic_data])

        assert pd.isnull(self.geometry_d['STATEFP']).sum() == 0
        good_idxs = self.geometry_d['STATEFP'].map(lambda x:x in FIPS_CODES_FOR_50_STATES_PLUS_DC).values
        print("Warning: the following State FIPS codes are being filtered out")
        print(self.geometry_d.loc[~good_idxs, 'STATEFP'].value_counts())
        print("%i/%i Census Block Groups in total removed" % ((~good_idxs).sum(), len(good_idxs)))
        self.geometry_d = self.geometry_d.loc[good_idxs]
        self.block_group_d = self.block_group_d.loc[good_idxs]
        self.geometry_d.index = self.geometry_d['GEOID_Data'].values
        self.block_group_d.index = self.block_group_d['GEOID'].values

    def annotate_with_income(self):
        """
        We want a single income number for each block group. This method computes that.
        """
        print("Computing household income")
        # copy-pasted column definitions right out of the codebook.
        codebook_string = """
        B19001e2    HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): Less than $10,000: Households -- (Estimate)
        B19001e3    HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $10,000 to $14,999: Households -- (Estimate)
        B19001e4    HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $15,000 to $19,999: Households -- (Estimate)
        B19001e5    HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $20,000 to $24,999: Households -- (Estimate)
        B19001e6    HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $25,000 to $29,999: Households -- (Estimate)
        B19001e7    HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $30,000 to $34,999: Households -- (Estimate)
        B19001e8    HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $35,000 to $39,999: Households -- (Estimate)
        B19001e9    HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $40,000 to $44,999: Households -- (Estimate)
        B19001e10   HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $45,000 to $49,999: Households -- (Estimate)
        B19001e11   HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $50,000 to $59,999: Households -- (Estimate)
        B19001e12   HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $60,000 to $74,999: Households -- (Estimate)
        B19001e13   HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $75,000 to $99,999: Households -- (Estimate)
        B19001e14   HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $100,000 to $124,999: Households -- (Estimate)
        B19001e15   HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $125,000 to $149,999: Households -- (Estimate)
        B19001e16   HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $150,000 to $199,999: Households -- (Estimate)
        B19001e17   HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $200,000 or more: Households -- (Estimate)
        """
        self.income_bin_edges = [0] + list(range(10000, 50000, 5000)) + [50000, 60000, 75000, 100000, 125000, 150000, 200000]

        income_column_names_to_vals = {}
        column_codes = codebook_string.split('\n')
        for f in column_codes:
            if len(f.strip()) == 0:
                continue
            col_name = f.split('HOUSEHOLD INCOME')[0].strip()
            if col_name == 'B19001e2':
                val = 10000
            elif col_name == 'B19001e17':
                val = 200000
            else:
                lower_bound = float(f.split('$')[1].split()[0].replace(',', ''))
                upper_bound = float(f.split('$')[2].split(':')[0].replace(',', ''))
                val = (lower_bound + upper_bound) / 2
            income_column_names_to_vals[col_name] = val
            print("The value for column %s is %2.1f" % (col_name, val))

        # each column gives the count of households with that income. So we need to take a weighted sum to compute the average income.
        self.block_group_d['total_household_income'] = 0.
        self.block_group_d['total_households'] = 0.
        for col in income_column_names_to_vals:
            self.block_group_d['total_household_income'] += self.block_group_d[col] * income_column_names_to_vals[col]
            self.block_group_d['total_households'] += self.block_group_d[col]
        self.block_group_d['mean_household_income'] = 1.*self.block_group_d['total_household_income'] / self.block_group_d['total_households']
        self.block_group_d['median_household_income'] = self.block_group_d['B19013e1'] # MEDIAN HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): Median household income in the past 12 months (in 2017 inflation-adjusted dollars): Households -- (Estimate)
        assert (self.block_group_d['total_households'] == self.block_group_d['B19001e1']).all() # sanity check: our count should agree with theirs.
        assert (pd.isnull(self.block_group_d['mean_household_income']) == (self.block_group_d['B19001e1'] == 0)).all()
        print("Warning: missing income data for %2.1f%% of census blocks with 0 households" % (pd.isnull(self.block_group_d['mean_household_income']).mean() * 100))
        self.income_column_names_to_vals = income_column_names_to_vals
        assert len(self.income_bin_edges) == len(self.income_column_names_to_vals)
        print(self.block_group_d[['mean_household_income', 'total_households']].describe())

    def annotate_with_counties_to_msa_mapping(self):
        """
        Annotate with metropolitan area info for consistency with Experienced Segregation paper.
        # https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2017/delineation-files/list1.xls
        """
        print("Loading county to MSA mapping")
        self.counties_to_msa_df = pd.read_csv(self.county_to_msa_mapping_filepath, skiprows=2, dtype={'FIPS State Code':str, 'FIPS County Code':str})
        print("%i rows read" % len(self.counties_to_msa_df))
        self.counties_to_msa_df = self.counties_to_msa_df[['CBSA Title',
                                                           'Metropolitan/Micropolitan Statistical Area',
                                                           'State Name',
                                                           'FIPS State Code',
                                                           'FIPS County Code']]

        self.counties_to_msa_df.columns = ['CBSA Title',
                                           'Metropolitan/Micropolitan Statistical Area',
                                           'State Name',
                                           'STATEFP',
                                           'COUNTYFP']

        self.counties_to_msa_df = self.counties_to_msa_df.dropna(how='all') # remove a couple blank rows.
        assert self.counties_to_msa_df['Metropolitan/Micropolitan Statistical Area'].map(lambda x:x in ['Metropolitan Statistical Area', 'Micropolitan Statistical Area']).all()
        print("Number of unique Metropolitan statistical areas: %i" %
            len(set(self.counties_to_msa_df.loc[self.counties_to_msa_df['Metropolitan/Micropolitan Statistical Area'] == 'Metropolitan Statistical Area', 'CBSA Title'])))
        print("Number of unique Micropolitan statistical areas: %i" %
            len(set(self.counties_to_msa_df.loc[self.counties_to_msa_df['Metropolitan/Micropolitan Statistical Area'] == 'Micropolitan Statistical Area', 'CBSA Title'])))
        old_len = len(self.geometry_d)
        assert len(self.counties_to_msa_df.drop_duplicates(['STATEFP', 'COUNTYFP'])) == len(self.counties_to_msa_df)


        self.geometry_d = self.geometry_d.merge(self.counties_to_msa_df,
                                                on=['STATEFP', 'COUNTYFP'],
                                                how='left')
        # For some reason the index gets reset here. Annoying, not sure why.
        self.geometry_d.index = self.geometry_d['GEOID_Data'].values

        assert len(self.geometry_d) == old_len
        assert (self.geometry_d.index == self.block_group_d.index).all()

    def get_demographic_stats_of_points(self, latitudes, longitudes, desired_cols):
        """
        Given a list or array of latitudes and longitudes, matches to Census Block Group.
        Returns a dictionary which includes the state and county FIPS code, along with any columns in desired_cols.

        This method assumes the latitudes and longitudes are in https://epsg.io/4326, which is what I think is used for Android/iOS -> SafeGraph coordinates.
        """
        def dtype_pandas_series(obj):
            return str(type(obj)) == "<class 'pandas.core.series.Series'>"
        assert not dtype_pandas_series(latitudes)
        assert not  dtype_pandas_series(longitudes)
        assert len(latitudes) == len(longitudes)

        t0 = time.time()

        # we have to match stuff a million rows at a time because otherwise we get weird memory warnings.
        start_idx = 0
        end_idx = start_idx + int(1e6)
        merged = []
        while start_idx < len(longitudes):
            print("Doing spatial join on points with indices from %i-%i" % (start_idx, min(end_idx, len(longitudes))))

            points = geopandas.GeoDataFrame(pd.DataFrame({'placeholder':np.array(range(start_idx, min(end_idx, len(longitudes))))}), # this column doesn't matter. We just have to create a geo data frame.
                geometry=geopandas.points_from_xy(longitudes[start_idx:end_idx], latitudes[start_idx:end_idx]),
                crs=self.crs_to_use)
            # see eg gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude)). http://geopandas.org/gallery/create_geopandas_from_pandas.html
            merged.append(sjoin(points, self.geometry_d[['geometry']], how='left', op='within'))
            assert len(merged[-1]) == len(points)
            start_idx += int(1e6)
            end_idx += int(1e6)
        merged = pd.concat(merged)
        merged.index = range(len(merged))
        assert list(merged.index) == list(merged['placeholder'])

        could_not_match = pd.isnull(merged['index_right']).values
        print("Cannot match to a CBG for a fraction %2.3f of points" % could_not_match.mean())

        results = {}
        for k in desired_cols + ['state_fips_code', 'county_fips_code', 'Metropolitan/Micropolitan Statistical Area', 'CBSA Title', 'GEOID_Data', 'TRACTCE']:
            results[k] = [None] * len(latitudes)
        results = pd.DataFrame(results)
        matched_geoids = merged['index_right'].values[~could_not_match]
        for c in desired_cols:
            results.loc[~could_not_match, c] = self.block_group_d.loc[matched_geoids, c].values
            if c in ['p_white', 'p_black', 'mean_household_income', 'median_household_income', 'new_census_monthly_rent_to_annual_income_multiplier', 'new_census_median_monthly_rent_to_annual_income_multiplier']:
                results[c] = results[c].astype('float')

        results.loc[~could_not_match, 'state_fips_code'] = self.geometry_d.loc[matched_geoids, 'STATEFP'].values
        results.loc[~could_not_match, 'county_fips_code'] = self.geometry_d.loc[matched_geoids, 'COUNTYFP'].values
        results.loc[~could_not_match, 'Metropolitan/Micropolitan Statistical Area'] = self.geometry_d.loc[matched_geoids,'Metropolitan/Micropolitan Statistical Area'].values
        results.loc[~could_not_match, 'CBSA Title'] = self.geometry_d.loc[matched_geoids, 'CBSA Title'].values
        results.loc[~could_not_match, 'GEOID_Data'] = self.geometry_d.loc[matched_geoids, 'GEOID_Data'].values
        results.loc[~could_not_match, 'TRACTCE'] = self.geometry_d.loc[matched_geoids, 'TRACTCE'].values

        print("Total query time is %2.3f" % (time.time() - t0))
        return results


def write_out_acs_5_year_data():
    #cbg_mapper = CensusBlockGroups(base_directory=PATH_FOR_CBG_MAPPER, gdb_files=None)
    cbg_mapper = CensusBlockGroups()

    geometry_cols = ['STATEFP',
              'COUNTYFP',
              'TRACTCE',
              'Metropolitan/Micropolitan Statistical Area',
              'CBSA Title',
              'State Name']
    block_group_cols = ['GEOID',
                              'p_black',
                              'p_white',
                              'p_asian',
                              'median_household_income',
                             'block_group_area_in_square_miles',
                             'people_per_mile']
    for k in geometry_cols:
        cbg_mapper.block_group_d[k] = cbg_mapper.geometry_d[k].values
    df_to_write_out = cbg_mapper.block_group_d[block_group_cols + geometry_cols]
    print("Total rows: %i" % len(df_to_write_out))
    print("Missing data")
    print(pd.isnull(df_to_write_out).mean())
    df_to_write_out.to_csv(PATH_TO_ACS_5YR_DATA)
