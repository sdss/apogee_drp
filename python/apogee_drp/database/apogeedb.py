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
import psycopg2 as pg
from psycopg2.extras import execute_values
import datetime

# type code
#  16  bool
#  18  char
#  20  int8, bigint
#  21  int2, smallint
#  23  int4, integer
#  25  text
# 700  float4, float
# 701  float8, double

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
def addapt_np_uint64(np_uint64):
    return AsIs(np_uint64)
def addapt_np_bool(np_bool):
    return AsIs(np_bool)
register_adapter(np.float16, addapt_np_float16)
register_adapter(np.float32, addapt_np_float32)
register_adapter(np.float64, addapt_np_float64)
register_adapter(np.int8, addapt_np_int8)
register_adapter(np.int16, addapt_np_int16)
register_adapter(np.int32, addapt_np_int32)
register_adapter(np.int64, addapt_np_int64)
register_adapter(np.uint64, addapt_np_uint64)
register_adapter(np.bool, addapt_np_bool)
register_adapter(np.bool_, addapt_np_bool)

from psycopg2.extensions import register_type
def cast_date(value, cursor):
    return value
oids = (1082, 1114, 1184) 
new_type = pg.extensions.new_type(oids, "DATE", cast_date)
register_type(new_type) 
# Cast None's for bool
oids_bool = (16,)
def cast_bool_none(value, cursor):
    if value is None:
        return False
    return np.bool(value)
new_type = pg.extensions.new_type(oids_bool, "BOOL", cast_bool_none)
register_type(new_type)
# Cast None's for text/char
oids_text = (18,25)
def cast_text_none(value, cursor):
    if value is None:
        return ''
    return value
new_type = pg.extensions.new_type(oids_text, "TEXT", cast_text_none)
register_type(new_type)
# Cast None's for integers
oids_int = (20,21,23)
def cast_int_none(value, cursor):
    if value is None:
        return -9999
    return np.int(value)
new_type = pg.extensions.new_type(oids_int, "INT", cast_int_none)
register_type(new_type)
# Cast None's for floats
oids_float = (700,701)
def cast_float_none(value, cursor):
    if value is None:
        return -999999.
    return np.float(value)
new_type = pg.extensions.new_type(oids_float, "FLOAT", cast_float_none)
register_type(new_type)

#def cast_data(data):
#    """ Cast output sql query data to deal with None's."""
#
#    # Recast
#    out = [
#        tuple('nan' if isinstance(i, np.floating) and np.isnan(i) else i for i in t)
#        for t in list(data)
#    ]


def register_date_typecasters(connection):
    """
    Casts date and timestamp values to string, resolves issues with out of
    range dates (e.g. BC) which psycopg2 can't handle
    """

    def cast_date(value, cursor):
        return value

    cursor = connection.cursor()
    cursor.execute("SELECT NULL::date")
    date_oid = cursor.description[0][1]
    cursor.execute("SELECT NULL::timestamp")
    timestamp_oid = cursor.description[0][1]
    cursor.execute("SELECT NULL::timestamp with time zone")
    timestamptz_oid = cursor.description[0][1]
    oids = (date_oid, timestamp_oid, timestamptz_oid)
    #oids = (1082, 1114, 1184)
    new_type = psycopg2.extensions.new_type(oids, "DATE", cast_date)
    pg.extensions.register_type(new_type) 


class DBSession(object):

    def __init__(self,host='pipelines'):
        """ Initialize the database session object. The connection is opened."""
        self.host = host
        if host=='operations':
            self.dbhost = 'operations.sdss.org'
            self.dbuser = 'sdss'
        elif host=='pipelines':
            self.dbhost = 'pipelines.sdss.org'
            self.dbuser = os.getlogin()
        else:
            raise ValueError(str(host)+' not supported')
        self.open()

    def open(self):
        """ Open the database connection."""
        connection = pg.connect(user=self.dbuser,host=self.dbhost,
                                password="",port = "5432",database = "sdss5db")
        self.connection = connection

    def close(self):
        """ Close the database connection."""
        self.connection.close()

    def query(self,table=None,cols='*',where=None,groupby=None,sql=None,fmt='numpy',verbose=False):
        """
        Query the APOGEE DRP database.

        Parameters
        ----------
        table : str, optional
            Name of table to query.  Default is to use the apogee_drp schema, but
              table names with schema (e.g. catalogdb.gaia_dr2_source) can also be input.
              If the sql command is given directly, then this is not needed.
        cols : str, optional
            Comma-separated list of columns to return.  Default is "*", all columns.
        where : str, optional
            Constraints on the selection.
        groupby : str, optional
            Column to group data by.
        sql : str, optional
            Enter the SQL command directly.
        fmt : str, optional
            The output format:
              -numpy: numpy structured array (default)
              -table: astropy table
              -list: list of tuples, first row has column names
              -raw: raw output, list of tuples
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

        cat = db.query(sql='select * from apgoee_drp.visit as v join catalogdb.something as c on v.apogee_id=c.2mass_type')

        """

        cur = self.connection.cursor()

        # Simple table query
        if sql is None:

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
            # The column names from the select query
            selcolnames = [desc[0] for desc in cur.description]

            if len(data)==0:
                cur.close()
                return np.array([])

            # Return the raw results
            if fmt=='raw':
                cur.close()
                return data
    
            # Get table column names and data types
            cur.execute("select column_name,data_type from information_schema.columns where table_schema='"+schema+"' and table_name='"+tab+"'")
            head = cur.fetchall()
            cur.close()
            colnames = [h[0] for h in head]
            headdict = dict(head)

            # Return fmt="list" format
            if fmt=='list':
                data = [tuple(colnames)]+data
                cur.close()
                return data

            # Get numpy data types
            d2d = {'smallint':np.int, 'integer':np.int, 'bigint':np.int, 'real':np.float32, 'double precision':np.float64,
                   'text':(np.str,200),'char':(np.str,5),'timestamp':(np.str,50), 'timestamp with time zone':(np.str,50),
                   'timestamp without time zone':(np.str,50),'boolean':np.bool}
            dt = []
            for i,name in enumerate(selcolnames):
                if name in headdict.keys():
                    if headdict[name]=='ARRAY':
                        # Get number if elements and type from the data itself
                        shp = np.array(data[0][i]).shape
                        type1 = np.array(data[0][i]).dtype.type
                        dt.append( (name, type1, shp) )
                    else:
                        dt.append( (name, d2d[headdict[name]]) )
                else:
                    raise ValueError(name+' not in header')
            dtype = np.dtype(dt)

            # Convert to numpy structured array
            cat = np.zeros(len(data),dtype=dtype)
            cat[...] = data
            del(data)

        # SQL command input
        else:
            
            # Execute the command
            if verbose:
                print('CMD = '+sql)
            cur.execute(sql)
            data = cur.fetchall()
            ndata = len(data)

            if len(data)==0:
                cur.close()
                return np.array([])

            # Return the raw results
            if fmt=='raw':
                cur.close()
                return data
    
            # Return fmt="list" format
            if fmt=='list':
                colnames = [desc[0] for desc in cur.description]
                data = [tuple(colnames)]+data
                cur.close()
                return data

            # Get table column names and data types
            colnames = [desc[0] for desc in cur.description]
            colnames = np.array(colnames)
            # Fix duplicate column names
            cindex = dln.create_index(colnames)
            bd,nbd = dln.where(cindex['num']>1)
            for i in range(nbd):
                ind = cindex['index'][cindex['lo'][bd[i]]:cindex['hi'][bd[i]]+1]
                ind.sort()
                nind = len(ind)
                for j in np.arange(1,nind):
                    colnames[ind[j]] += str(j+1)

            # Use the data returned to get the type
            dt = []
            stringizecol = []
            for i,c in enumerate(colnames):
                type1 = type(data[0][i])
                if type1 is str:
                    dt.append( (c, type(data[0][i]), 300) )
                elif type1 is list:  # convert list to array
                    nlist = len(data[0][i])
                    dtype1 = type(data[0][i][0])
                    dt.append( (c, dtype1, nlist) )
                else:
                    dtype1 = type(data[0][i])
                    # Check for None/null, need a "real" value to get the type
                    if (data[0][i] is None):
                        cnt = 0
                        while (data[cnt][i] is None) and (cnt<(ndata-1)) and (cnt<10000): cnt += 1
                        if data[cnt][i] is not None:
                            dtype1 = type(data[cnt][i])
                        else:  # still None, use float
                            dtype1 = (str,300)  # changed from float to str, DLN 9/7/23
                            stringizecol.append(i)
                    dt.append( (c, dtype1) )
            dtype = np.dtype(dt)

            # Convert columns for which we couldn't figure out the data type to strings
            if len(stringizecol)>0:
                for i in range(len(data)):
                    data1 = list(data[i])  # convert to list first so we can modify it
                    for k in range(len(stringizecol)):
                        data1[stringizecol[k]] = str(data1[stringizecol[k]])
                    data[i] = tuple(data1)  # convert back to tuple for numpy array loading
            
            # Convert to numpy structured array
            cat = np.zeros(len(data),dtype=dtype)
            cat[...] = data
            del(data)


        # For string columns change size to maximum length of that column
        dt2 = []
        names = dtype.names
        nplen = np.vectorize(len)
        needcopy = False
        for i in range(len(dtype)):
            type1 = type(cat[names[i]][0])
            if type1 is str or type1 is np.str_:
                maxlen = np.max(nplen(cat[names[i]]))
                dt2.append( (names[i], str, maxlen+10) )
                needcopy = True
            else:
                dt2.append(dt[i])  # reuse dt value
        # We need to copy
        if needcopy==True:
            dtype2 = np.dtype(dt2)
            cat2 = np.zeros(len(cat),dtype=dtype2)
            for n in names:
                cat2[n] = cat[n]
            cat = cat2
            del cat2

        # Convert to astropy table
        if fmt=='table':
            cat = Table(cat)
                    
        return cat


    def ingest(self,table,cat,onconflict='update',constraintname=None,verbose=False):
        """
        Insert/ingest data into the database.

        Parameters
        ----------
        table : str
            Name of table to query.  Default is to use the apogee_drp schema, but
              table names with schema (e.g. catalogdb.gaia_dr2_source) can also be input.
        cat : numpy structured array or astropy table
            Catalog as (1) numpy structured array or (2) astropy table to insert into db table.
        onconflict: str, optional
            What to do when there is a uniqueness requirement on the table and there is
              a conflict (i.e. one of the inserted rows will create a duplicate).  The
              options are:
              'update': update the existing row with the information from the new insert (default).
              'nothing': do nothing, leave the existing row as is and do not insert the
                          new conflicting row.
        constraintname : str, optional
            If onconflict='update', then this should be the name of the unique columns
              (comma-separated list of column names).
        verbose : bool, optional
            Verbose output to screen.

        Returns
        -------
        The catalog is inserted into the database table.
        Nothing is returned.

        Examples
        --------
        db.ingest('visit',cat)

        """

        ncat = dln.size(cat)
        cur = self.connection.cursor()

        # Convert astropy table to numpy structured array
        if isinstance(cat,Table):
            cat = np.array(cat)

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

        # Check for arrays and replace with a list
        hasarrays = False
        for d in data[0]:
            hasarrays |= hasattr(d,'__len__') and type(d) is not str
        if hasarrays:
            data1 = data.copy()
            data = [
                tuple(i.tolist() if  hasattr(i,'__len__') and type(i) is not str and type(i) is not np.str_ else i for i in t)
                for t in list(data1)
            ]
            del data1
            
        
        # On conflict do nothing
        if onconflict=='nothing':
            insert_query = 'INSERT INTO '+schema+'.'+tab+' ('+','.join(columns)+') VALUES %s ON CONFLICT DO NOTHING'
        # On conflict do update
        elif onconflict=='update':
            # Get the unique constraint from database
            if constraintname is None:
                cur.execute("select conname from pg_constraint where contype='u' and conrelid='"+schema+"."+tab+"'::regclass::oid")
                out = cur.fetchall()
                if len(out)>1:
                    raise Exception('More than ONE unique constraint found for '+schema+'.'+tab)
                if len(out)>0:
                    constraintname = out[0][0]
                    constraintstr = 'ON CONSTRAINT '+constraintname
                else:
                    # No constraint
                    constraintname = None
            # Constraint column names input
            else:
                constraintstr = '('+constraintname+')'
            # There is a constraint, use on conflict to update
            if constraintname is not None:
                excluded = ','.join(['"'+c+'"=excluded."'+c+'"' for c in columns])
                # Include CREATED in the list of columns to update (if it exists in the table)
                #  get table column names
                cur.execute("select column_name,data_type from information_schema.columns where table_schema='"+schema+"' and table_name='"+tab+"'")
                head = cur.fetchall()
                tabcolnames = [h[0] for h in head]
                if 'created' in tabcolnames:
                    excluded += ',"created"=now()'
                # Create the insert statement
                insert_query = 'INSERT INTO '+schema+'.'+tab+' ('+','.join(columns)+') VALUES %s ON CONFLICT '+constraintstr+\
                               ' DO UPDATE SET '+excluded
            else: # no constraint
                insert_query = 'INSERT INTO '+schema+'.'+tab+' ('+','.join(columns)+') VALUES %s ON CONFLICT DO NOTHING'
        else:
            raise ValueError(onconflict+' not supported')
        # Perform the insert
        execute_values(cur,insert_query,data,template=None)

        self.connection.commit()
        cur.close()

        if verbose:
            print(str(len(cat))+' rows inserted into '+schema+'.'+tab)


    def update(self,table,cat,verbose=False):
        """
        Update values in a database table.

        Parameters
        ----------
        table : str
            Name of table to query.  Default is to use the apogee_drp schema, but
              table names with schema (e.g. catalogdb.gaia_dr2_source) can also be input.
        cat : numpy structured array or astropy table
            Catalog as (1) numpy structured array or (2) astropy table to insert into db
              table. The first column must be a unique ID or key.
        verbose : bool, optional
            Verbose output to screen.

        Returns
        -------
        The values are updated in the table.
        Nothing is returned.

        Examples
        --------
        db.update('visit',cat)

        """

        ncat = dln.size(cat)
        cur = self.connection.cursor()

        # Convert astropy table to numpy structured array
        if isinstance(cat,Table):
            cat = np.array(cat)

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

        # Check for arrays and replace with a list
        hasarrays = False
        for d in data[0]:
            hasarrays |= hasattr(d,'__len__') and type(d) is not str
        if hasarrays:
            data1 = data.copy()
            data = [
                tuple(i.tolist() if  hasattr(i,'__len__') and type(i) is not str and type(i) is not np.str_ else i for i in t)
                for t in list(data1)
            ]
            del data1
            
        setcmd = ','.join([c+'=d.'+c for c in columns[1:]])
        update_query = 'UPDATE '+schema+'.'+tab+' AS t SET '+setcmd+' FROM (VALUES %s) '+\
                       ' AS d ('+','.join(columns)+') WHERE t.'+columns[0]+'=d.'+columns[0]
        execute_values(cur,update_query,data,template=None)

        self.connection.commit()
        cur.close()

        if verbose:
            print(str(len(cat))+' rows updated in '+schema+'.'+tab)


    def delete(self,table,cat,verbose=False):
        """
        Delete values in a database table.

        Parameters
        ----------
        table : str
            Name of table to query.  Default is to use the apogee_drp schema, but
              table names with schema (e.g. catalogdb.gaia_dr2_source) can also be input.
        cat : numpy structured array or astropy table
            Catalog as (1) numpy structured array or (2) astropy table that contains
             unique ID/keys of rows to delete.  Only the first column is used.
        verbose : bool, optional
            Verbose output to screen.

        Returns
        -------
        The values are deleted in the table.
        Nothing is returned.

        Examples
        --------
        db.delete('visit',cat)

        """

        ncat = dln.size(cat)
        cur = self.connection.cursor()

        # Convert astropy table to numpy structured array
        if isinstance(cat,Table):
            cat = np.array(cat)

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

        keyname = cat.dtype.names[0]
        data = [(d,) for d in cat[keyname]]
        delete_query = "DELETE FROM "+schema+"."+tab+" WHERE "+keyname+" in (%s)"
        execute_values(cur,delete_query,data,template=None)

        self.connection.commit()
        cur.close()

        if verbose:
            print(str(len(cat))+' rows deleted from '+schema+'.'+tab)

