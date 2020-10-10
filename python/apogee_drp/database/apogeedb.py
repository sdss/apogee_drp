#!/usr/bin/env python

# Fit proper motion and parallax using ra/dec/mjd data

# Most of this code was taken from here:
# https://github.com/ctheissen/WISE_Parallaxes/blob/master/WISE_Parallax.py

import os, sys
import numpy as np
from astropy.table import Table, vstack, join
#import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time
import astropy.coordinates as coords
from dlnpyutils import utils as dln, coords as dcoords
import time
import psycopg2 as pq
from psycopg2.extras import execute_values


from psycopg2.extensions import register_adapter, AsIs
def addapt_np_float16(np_float16):
    return AsIs(np_float16)
def addapt_np_float32(np_float32):
    return AsIs(np_float32)
def addapt_np_float64(np_float64):
    return AsIs(np_float64)
def addapt_np_int8(np_int8):
    return AsIs(np_int8)
def addapt_np_int16(np_int16):
    return AsIs(np_int16)
def addapt_np_int32(np_int32):
    return AsIs(np_int32)
def addapt_np_int64(np_int64):
    return AsIs(np_int64)
register_adapter(np.float16, addapt_np_float16)
register_adapter(np.float32, addapt_np_float32)
register_adapter(np.float64, addapt_np_float64)
register_adapter(np.int8, addapt_np_int8)
register_adapter(np.int16, addapt_np_int16)
register_adapter(np.int32, addapt_np_int32)
register_adapter(np.int64, addapt_np_int64)

class DBSession(object):

    def __init__(self):
        """ Initialize the database session object. The connection is opened."""
        self.open()

    def open(self):
        """ Open the database connection."""
        connection = pq.connect(user="sdss",host="operations.sdss.org",
                                password="",port = "5432",database = "sdss5db")
        self.connection = connection

    def close(self):
        """ Close the database connection."""
        self.connection.close()

    def query(self,table,cols='*',where=None,groupby=None,raw=False,verbose=False):
        """
        Query the APOGEE DRP database.

        Parameters
        ----------
        table : str
            Name of table to query.  Default is to use the apogee_drp schema, but
              table names with schema (e.g. catalogdb.gaia_dr2_source) can also be input.
        cols : str, optional
            Comma-separated list of columns to return.  Default is "*", all columns.
        where : str, optional
            Constraints on the selection.
        groupby : str, optional
            Column to group data by.
        raw : bool, optional
            Return the raw database output.  The default is to return the
              data as a numpy structured array.
        verbose : bool, optional
            Print verbose output to screen.  False by default.

        Returns
        -------
        cat : numpy structured array
           The data in a catalog format.  If raw=True then the data will be returned
            as a list of tuples.

        Examples
        --------
        cat = db.query('visit',where="apogee_id='2M09241296+2723318'")

        """

        cur = self.connection.cursor()

        # Schema
        if table.find('.')>-1:
            schema,tab = table.split('.')
        else:
            schema = 'apogee_drp'
            tab = table

        # Start the SELECT statement
        cmd = 'SELECT '+cols+' FROM '+schema+'.'+tab

        # Add WHERE statement
        if where is not None:
            cmd += ' WHERE '+where

        # Add GROUP BY statement
        if groupby is not None:
            cmd += ' GROUP BY '+groupby
        
        # Execute the select command
        if verbose:
            print('CMD = '+cmd)
        cur.execute(cmd)
        data = cur.fetchall()

        if len(data)==0:
            cur.close()
            return np.array([])

        # Return the raw results
        if raw is True:
            cur.close()
            return data
    
        # Get table column names and data types
        cur.execute("select column_name,data_type from information_schema.columns where table_schema='"+schema+"' and table_name='"+tab+"'")
        head = cur.fetchall()
        cur.close()

        d2d = {'smallint':np.int, 'integer':np.int, 'bigint':np.int, 'real':np.float32, 'double precision':np.float64,
               'text':(np.str,200),'char':(np.str,5)}
        colnames = [h[0] for h in head]
        dt = []
        for h in head:
            dt.append( (h[0], d2d[h[1]]) )
        dtype = np.dtype(dt)

        # Convert to numpy structured array
        cat = np.zeros(len(data),dtype=dtype)
        cat[...] = data
        del(data)

        return cat


    def load(self,table,cat,verbose=False):
        """
        Load a catalog into the database.

        Parameters
        ----------
        table : str
            Name of table to query.  Default is to use the apogee_drp schema, but
              table names with schema (e.g. catalogdb.gaia_dr2_source) can also be input.
        cat : numpy structured array
            Catalog as numpy structured array to insert into table.
        verbose : bool, optional
            Verbose output to screen.

        Returns
        -------
        The catalog is inserted into the database table.
        Nothing is returned.

        Examples
        --------
        db.load('visit',cat)

        """

        ncat = dln.size(cat)
        cur = self.connection.cursor()

        # Schema
        if table.find('.')>-1:
            schema,tab = table.split('.')
        else:
            schema = 'apogee_drp'
            tab = table

        # Make sure the table already exists
        cur.execute("select table_name from information_schema.tables where table_schema='"+schema+"'")
        qtabs = cur.fetchall()
        alltabs = [q[0] for q in qtabs]
        if tab not in alltabs:
            raise Exception(tab+' table not in '+schema+' schema')

        # Get the column names
        cnames = cat.dtype.names
        cdict = dict(cat.dtype.fields)
        # Insert statement
        columns = [n.lower() for n in cnames]
        
        # Replace nan with 'nan'  
        data = [
            tuple('nan' if isinstance(i, np.floating) and np.isnan(i) else i for i in t)
            for t in list(cat)
        ]

        insert_query = 'INSERT INTO '+schema+'.'+tab+' ('+','.join(columns)+') VALUES %s'
        execute_values(cur,insert_query,data,template=None)

        self.connection.commit()
        cur.close()

        if verbose:
            print(str(len(cat))+' rows inserted into '+schema+'.'+tab)
