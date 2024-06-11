import numpy as np

data_predict = dict(
    SPX=dict(
        SPX=dict(
            source='YAHOO',
            yahoo_ticker='^GSPC',
            transform=np.log
        ),
        CPIAUCNS=dict(
            source='FRED',
            freq_adj='M',
            transform=np.log
        ),
        FEDFUNDS=dict(
            source='FRED',
            freq_adj='M'
        ),
        WALCL=dict(
            source='FRED',
            freq_adj='NA'
        ),
        GDP=dict(
            source='FRED',
            freq_adj='Q',
            transform=np.log
        ),
        INDPRO=dict(
            source='FRED',
            freq_adj='M',
            transform=np.log
        ),
        LES1252881600Q=dict(
            source='FRED',
            freq_adj='Q',
            transform=np.log
        ),
        M2NS=dict(
            source='FRED',
            freq_adj='M',
            transform=np.log
        ),
        T10Y2Y=dict(
            source='FRED',
            freq_adj='NA'
        ),
        UMCSENT=dict(
            source='FRED',
            freq_adj='M',
            transform=np.log
        ),
        PE_RATIO=dict(
            source='MULTPL',
            name='s-p-500-pe-ratio',
            freq_adj='M'
        ),
        SCHILLER_PE=dict(
            source='MULTPL',
            name='shiller-pe',
            freq_adj='M'
        ),
        #PB_RATIO=dict(
        #    source='MULTPL',
        #    name='s-p-500-pb-ratio',
        #    freq_adj='M'
        #)
    ),
    USDMXN=dict(
        USDMXN=dict(
            source='BANXICO',
            bmx_serie='SF43718',
            freq_adj='NA',
            transform=np.log
        ),
        USDMXN_LOW=dict(
            source='YAHOO',
            yahoo_ticker='MXN=X',
            col='Low'
        ),
        USDMXN_HIGH=dict(
            source='YAHOO',
            yahoo_ticker='MXN=X',
            col='High'
        ),
        DCOILWTICO=dict(
            source='FRED',
            freq_adj='NA',
        ),
        CPIAUCNS=dict(
            source='FRED',
            freq_adj='M',
            transform=np.log
        ),
        INPC=dict(
            source='BANXICO',
            bmx_serie='SP1',
            freq_adj='M',
            transform=np.log
        ),
        FEDFUNDS=dict(
            source='FRED',
            freq_adj='M'
        ),
        CETES28=dict(
            source='BANXICO',
            bmx_serie='SF43936',
            freq_adj='NA'
        ),
        CUENTA_CORRIENTE=dict(
            source='BANXICO',
            bmx_serie='SE44352',
            freq_adj='Q'
        ),
        CUENTA_CAPITAL=dict(
            source='BANXICO',
            bmx_serie='SE44393',
            freq_adj='Q'
        ),
        SPX=dict(
            source='YAHOO',
            yahoo_ticker='^GSPC',
            transform=np.log
        ),
        DGS10=dict(
            source='FRED',
            freq_adj='NA'
        )
        #CUENTA_FINANCIERA=dict(
        #    source='BANXICO',
        #    bmx_serie='SE44396',
        #    freq_adj='Q'
        #),
        #VAR_RESERVA_INT=dict(
        #    source='BANXICO',
        #    bmx_serie='SE44458',
        #    freq_adj='Q'
        #),
        #DEUDA_TOTAL=dict(
        #    source='BANXICO',
        #    bmx_serie='SG193',
        #    freq_adj='M',
        #    transform=np.log
        #),
        #DEUDA_EXTERNA=dict(
        #    source='BANXICO',
        #   bmx_serie='SG195',
        #    freq_adj='M',
        #   transform=np.log
        #),
        #DTWEXAFEGS=dict(
        #    source='FRED',
        #    freq_adj='NA',
        #    transform=np.log
        #),
        #T10Y2Y=dict(
        #    source='FRED',
        #    freq_adj='NA'
        #),
        #UMCSENT=dict(
        #    source='FRED',
        #    freq_adj='M',
        #    transform=np.log
        #),
        #GDP=dict(
        #    source='FRED',
        #    freq_adj='Q',
        #    transform=np.log
        #),
        #ACT_ECON=dict(
        #    source='INEGI',
        #    series='496150',
        #    transform=np.log
        #),
        #IND_ADEL=dict(
        #    source='INEGI',
        #    series='214307',
        #),
        #REMESAS=dict(
        #    source='BANXICO',
        #    bmx_serie='SE27803',
        #    freq_adj='M',
        #)
    )
)
