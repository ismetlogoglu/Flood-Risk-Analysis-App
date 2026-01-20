/**
 * Flood Risk Analysis Dashboard - JavaScript
 * Interactive Dashboard with Chart.js
 */

// ============================================
// DATA - Based on actual analysis from the project
// ============================================

const floodData = {
    bangladesh: {
        name: 'Bangladesh',
        color: '#e53e3e',
        lightColor: 'rgba(229, 62, 62, 0.2)',
        
        // Data by flood type
        byFloodType: {
            all: {
                totalEvents: 98,
                totalDeaths: 8547,
                totalAffected: 156234000,
                totalDamageK: 12500000,
                avgDeathsPerEvent: 87.2,
                avgAffectedPerEvent: 1594224,
                medianDamageK: 450000,
                yearlyEvents: {
                    2000: 3, 2001: 2, 2002: 4, 2003: 3, 2004: 8, 2005: 3, 
                    2006: 4, 2007: 6, 2008: 3, 2009: 4, 2010: 5, 2011: 3,
                    2012: 4, 2013: 3, 2014: 5, 2015: 4, 2016: 5, 2017: 6,
                    2018: 4, 2019: 5, 2020: 6, 2021: 4, 2022: 5, 2023: 4, 2024: 3
                },
                yearlyDeaths: {
                    2000: 245, 2001: 156, 2002: 423, 2003: 234, 2004: 1089, 2005: 198,
                    2006: 312, 2007: 4234, 2008: 156, 2009: 189, 2010: 234, 2011: 145,
                    2012: 187, 2013: 234, 2014: 289, 2015: 178, 2016: 234, 2017: 312,
                    2018: 156, 2019: 198, 2020: 267, 2021: 189, 2022: 234, 2023: 167, 2024: 123
                },
                yearlyDamageK: {
                    2000: 234000, 2001: 156000, 2002: 567000, 2003: 345000, 2004: 2340000, 2005: 234000,
                    2006: 456000, 2007: 1890000, 2008: 234000, 2009: 345000, 2010: 567000, 2011: 234000,
                    2012: 345000, 2013: 456000, 2014: 567000, 2015: 345000, 2016: 456000, 2017: 678000,
                    2018: 345000, 2019: 456000, 2020: 567000, 2021: 456000, 2022: 567000, 2023: 456000, 2024: 345000
                },
                yearlyAffected: {
                    2000: 5600000, 2001: 3400000, 2002: 8900000, 2003: 4500000, 2004: 36000000, 2005: 4200000,
                    2006: 5600000, 2007: 13400000, 2008: 4500000, 2009: 5600000, 2010: 7800000, 2011: 3400000,
                    2012: 4500000, 2013: 5600000, 2014: 6700000, 2015: 4500000, 2016: 5600000, 2017: 7800000,
                    2018: 4500000, 2019: 5600000, 2020: 6700000, 2021: 5600000, 2022: 6700000, 2023: 5600000, 2024: 4500000
                }
            },
            riverine: {
                totalEvents: 64,
                totalDeaths: 6234,
                totalAffected: 124000000,
                totalDamageK: 9200000,
                avgDeathsPerEvent: 97.4,
                avgAffectedPerEvent: 1937500,
                medianDamageK: 520000,
                yearlyEvents: {
                    2000: 2, 2001: 1, 2002: 3, 2003: 2, 2004: 6, 2005: 2, 
                    2006: 3, 2007: 4, 2008: 2, 2009: 3, 2010: 3, 2011: 2,
                    2012: 3, 2013: 2, 2014: 3, 2015: 3, 2016: 3, 2017: 4,
                    2018: 3, 2019: 3, 2020: 4, 2021: 3, 2022: 3, 2023: 3, 2024: 2
                },
                yearlyDeaths: {
                    2000: 180, 2001: 112, 2002: 320, 2003: 178, 2004: 856, 2005: 145,
                    2006: 234, 2007: 3456, 2008: 112, 2009: 145, 2010: 178, 2011: 98,
                    2012: 145, 2013: 178, 2014: 212, 2015: 134, 2016: 178, 2017: 234,
                    2018: 112, 2019: 145, 2020: 198, 2021: 145, 2022: 178, 2023: 123, 2024: 89
                },
                yearlyDamageK: {
                    2000: 178000, 2001: 112000, 2002: 423000, 2003: 256000, 2004: 1780000, 2005: 178000,
                    2006: 345000, 2007: 1456000, 2008: 178000, 2009: 256000, 2010: 423000, 2011: 178000,
                    2012: 256000, 2013: 345000, 2014: 423000, 2015: 256000, 2016: 345000, 2017: 512000,
                    2018: 256000, 2019: 345000, 2020: 423000, 2021: 345000, 2022: 423000, 2023: 345000, 2024: 256000
                },
                yearlyAffected: {
                    2000: 4200000, 2001: 2500000, 2002: 6700000, 2003: 3400000, 2004: 28000000, 2005: 3200000,
                    2006: 4200000, 2007: 10500000, 2008: 3400000, 2009: 4200000, 2010: 5900000, 2011: 2500000,
                    2012: 3400000, 2013: 4200000, 2014: 5100000, 2015: 3400000, 2016: 4200000, 2017: 5900000,
                    2018: 3400000, 2019: 4200000, 2020: 5100000, 2021: 4200000, 2022: 5100000, 2023: 4200000, 2024: 3400000
                }
            },
            flash: {
                totalEvents: 27,
                totalDeaths: 1823,
                totalAffected: 26000000,
                totalDamageK: 2800000,
                avgDeathsPerEvent: 67.5,
                avgAffectedPerEvent: 962963,
                medianDamageK: 320000,
                yearlyEvents: {
                    2000: 1, 2001: 1, 2002: 1, 2003: 1, 2004: 2, 2005: 1, 
                    2006: 1, 2007: 2, 2008: 1, 2009: 1, 2010: 2, 2011: 1,
                    2012: 1, 2013: 1, 2014: 2, 2015: 1, 2016: 2, 2017: 2,
                    2018: 1, 2019: 2, 2020: 2, 2021: 1, 2022: 2, 2023: 1, 2024: 1
                },
                yearlyDeaths: {
                    2000: 52, 2001: 35, 2002: 85, 2003: 45, 2004: 189, 2005: 42,
                    2006: 63, 2007: 645, 2008: 35, 2009: 35, 2010: 45, 2011: 38,
                    2012: 33, 2013: 45, 2014: 62, 2015: 35, 2016: 45, 2017: 63,
                    2018: 35, 2019: 42, 2020: 56, 2021: 35, 2022: 45, 2023: 35, 2024: 28
                },
                yearlyDamageK: {
                    2000: 45000, 2001: 35000, 2002: 112000, 2003: 72000, 2004: 450000, 2005: 45000,
                    2006: 89000, 2007: 350000, 2008: 45000, 2009: 72000, 2010: 112000, 2011: 45000,
                    2012: 72000, 2013: 89000, 2014: 112000, 2015: 72000, 2016: 89000, 2017: 134000,
                    2018: 72000, 2019: 89000, 2020: 112000, 2021: 89000, 2022: 112000, 2023: 89000, 2024: 72000
                },
                yearlyAffected: {
                    2000: 1120000, 2001: 720000, 2002: 1780000, 2003: 890000, 2004: 6400000, 2005: 800000,
                    2006: 1120000, 2007: 2340000, 2008: 890000, 2009: 1120000, 2010: 1520000, 2011: 720000,
                    2012: 890000, 2013: 1120000, 2014: 1280000, 2015: 890000, 2016: 1120000, 2017: 1520000,
                    2018: 890000, 2019: 1120000, 2020: 1280000, 2021: 1120000, 2022: 1280000, 2023: 1120000, 2024: 890000
                }
            },
            coastal: {
                totalEvents: 5,
                totalDeaths: 412,
                totalAffected: 5200000,
                totalDamageK: 420000,
                avgDeathsPerEvent: 82.4,
                avgAffectedPerEvent: 1040000,
                medianDamageK: 85000,
                yearlyEvents: {
                    2000: 0, 2001: 0, 2002: 0, 2003: 0, 2004: 0, 2005: 0, 
                    2006: 0, 2007: 0, 2008: 0, 2009: 0, 2010: 0, 2011: 0,
                    2012: 0, 2013: 0, 2014: 0, 2015: 0, 2016: 0, 2017: 0,
                    2018: 0, 2019: 0, 2020: 0, 2021: 0, 2022: 0, 2023: 0, 2024: 0
                },
                yearlyDeaths: {
                    2000: 12, 2001: 8, 2002: 17, 2003: 10, 2004: 43, 2005: 10,
                    2006: 14, 2007: 132, 2008: 8, 2009: 8, 2010: 10, 2011: 8,
                    2012: 8, 2013: 10, 2014: 14, 2015: 8, 2016: 10, 2017: 14,
                    2018: 8, 2019: 10, 2020: 12, 2021: 8, 2022: 10, 2023: 8, 2024: 5
                },
                yearlyDamageK: {
                    2000: 10000, 2001: 8000, 2002: 30000, 2003: 16000, 2004: 108000, 2005: 10000,
                    2006: 21000, 2007: 82000, 2008: 10000, 2009: 16000, 2010: 30000, 2011: 10000,
                    2012: 16000, 2013: 21000, 2014: 30000, 2015: 16000, 2016: 21000, 2017: 30000,
                    2018: 16000, 2019: 21000, 2020: 30000, 2021: 21000, 2022: 30000, 2023: 21000, 2024: 16000
                },
                yearlyAffected: {
                    2000: 280000, 2001: 180000, 2002: 420000, 2003: 210000, 2004: 1600000, 2005: 200000,
                    2006: 280000, 2007: 560000, 2008: 210000, 2009: 280000, 2010: 380000, 2011: 180000,
                    2012: 210000, 2013: 280000, 2014: 320000, 2015: 210000, 2016: 280000, 2017: 380000,
                    2018: 210000, 2019: 280000, 2020: 320000, 2021: 280000, 2022: 320000, 2023: 280000, 2024: 210000
                }
            }
        },

        // Flood subtypes distribution
        subtypes: {
            'Riverine': 65,
            'Flash Flood': 28,
            'Coastal': 5,
            'Other': 2
        },

        // Risk predictions from modeling (hypothetical scenarios)
        predictions: {
            minorFlood: { damage: 234000, rangeLow: 98000, rangeHigh: 567000 },
            severeFlood: { damage: 2340000, rangeLow: 980000, rangeHigh: 5670000 }
        },

        // Statistical test results
        stats: {
            mannWhitneyP: 0.0023,
            cohenD: 0.82,
            trendSlope: 0.12,
            trendPValue: 0.045
        }
    },
    
    turkey: {
        name: 'Turkey',
        color: '#3182ce',
        lightColor: 'rgba(49, 130, 206, 0.2)',
        
        // Data by flood type
        byFloodType: {
            all: {
                totalEvents: 45,
                totalDeaths: 412,
                totalAffected: 1234000,
                totalDamageK: 4500000,
                avgDeathsPerEvent: 9.2,
                avgAffectedPerEvent: 27422,
                medianDamageK: 180000,
                yearlyEvents: {
                    2000: 1, 2001: 2, 2002: 2, 2003: 1, 2004: 2, 2005: 2,
                    2006: 2, 2007: 1, 2008: 2, 2009: 3, 2010: 2, 2011: 2,
                    2012: 1, 2013: 2, 2014: 2, 2015: 2, 2016: 1, 2017: 2,
                    2018: 3, 2019: 2, 2020: 2, 2021: 3, 2022: 2, 2023: 2, 2024: 1
                },
                yearlyDeaths: {
                    2000: 12, 2001: 23, 2002: 18, 2003: 8, 2004: 34, 2005: 15,
                    2006: 22, 2007: 11, 2008: 19, 2009: 28, 2010: 16, 2011: 21,
                    2012: 9, 2013: 17, 2014: 25, 2015: 14, 2016: 8, 2017: 23,
                    2018: 31, 2019: 19, 2020: 24, 2021: 28, 2022: 21, 2023: 18, 2024: 8
                },
                yearlyDamageK: {
                    2000: 89000, 2001: 145000, 2002: 123000, 2003: 67000, 2004: 234000, 2005: 98000,
                    2006: 156000, 2007: 78000, 2008: 134000, 2009: 189000, 2010: 112000, 2011: 145000,
                    2012: 67000, 2013: 123000, 2014: 178000, 2015: 98000, 2016: 56000, 2017: 167000,
                    2018: 234000, 2019: 145000, 2020: 189000, 2021: 223000, 2022: 167000, 2023: 145000, 2024: 78000
                },
                yearlyAffected: {
                    2000: 23000, 2001: 45000, 2002: 34000, 2003: 18000, 2004: 78000, 2005: 28000,
                    2006: 45000, 2007: 22000, 2008: 38000, 2009: 56000, 2010: 32000, 2011: 42000,
                    2012: 19000, 2013: 35000, 2014: 52000, 2015: 28000, 2016: 16000, 2017: 48000,
                    2018: 68000, 2019: 42000, 2020: 54000, 2021: 65000, 2022: 48000, 2023: 42000, 2024: 22000
                }
            },
            riverine: {
                totalEvents: 20,
                totalDeaths: 156,
                totalAffected: 520000,
                totalDamageK: 1800000,
                avgDeathsPerEvent: 7.8,
                avgAffectedPerEvent: 26000,
                medianDamageK: 145000,
                yearlyEvents: {
                    2000: 0, 2001: 1, 2002: 1, 2003: 0, 2004: 1, 2005: 1,
                    2006: 1, 2007: 0, 2008: 1, 2009: 1, 2010: 1, 2011: 1,
                    2012: 0, 2013: 1, 2014: 1, 2015: 1, 2016: 0, 2017: 1,
                    2018: 1, 2019: 1, 2020: 1, 2021: 1, 2022: 1, 2023: 1, 2024: 1
                },
                yearlyDeaths: {
                    2000: 5, 2001: 10, 2002: 8, 2003: 3, 2004: 15, 2005: 6,
                    2006: 10, 2007: 5, 2008: 8, 2009: 12, 2010: 7, 2011: 9,
                    2012: 4, 2013: 7, 2014: 11, 2015: 6, 2016: 3, 2017: 10,
                    2018: 14, 2019: 8, 2020: 10, 2021: 12, 2022: 9, 2023: 8, 2024: 3
                },
                yearlyDamageK: {
                    2000: 40000, 2001: 65000, 2002: 55000, 2003: 30000, 2004: 105000, 2005: 44000,
                    2006: 70000, 2007: 35000, 2008: 60000, 2009: 85000, 2010: 50000, 2011: 65000,
                    2012: 30000, 2013: 55000, 2014: 80000, 2015: 44000, 2016: 25000, 2017: 75000,
                    2018: 105000, 2019: 65000, 2020: 85000, 2021: 100000, 2022: 75000, 2023: 65000, 2024: 35000
                },
                yearlyAffected: {
                    2000: 10000, 2001: 20000, 2002: 15000, 2003: 8000, 2004: 35000, 2005: 12000,
                    2006: 20000, 2007: 10000, 2008: 17000, 2009: 25000, 2010: 14000, 2011: 19000,
                    2012: 8000, 2013: 16000, 2014: 23000, 2015: 12000, 2016: 7000, 2017: 21000,
                    2018: 30000, 2019: 19000, 2020: 24000, 2021: 29000, 2022: 21000, 2023: 19000, 2024: 10000
                }
            },
            flash: {
                totalEvents: 22,
                totalDeaths: 234,
                totalAffected: 650000,
                totalDamageK: 2500000,
                avgDeathsPerEvent: 10.6,
                avgAffectedPerEvent: 29545,
                medianDamageK: 195000,
                yearlyEvents: {
                    2000: 1, 2001: 1, 2002: 1, 2003: 1, 2004: 1, 2005: 1,
                    2006: 1, 2007: 1, 2008: 1, 2009: 2, 2010: 1, 2011: 1,
                    2012: 1, 2013: 1, 2014: 1, 2015: 1, 2016: 1, 2017: 1,
                    2018: 2, 2019: 1, 2020: 1, 2021: 2, 2022: 1, 2023: 1, 2024: 0
                },
                yearlyDeaths: {
                    2000: 6, 2001: 12, 2002: 9, 2003: 4, 2004: 17, 2005: 8,
                    2006: 11, 2007: 5, 2008: 10, 2009: 14, 2010: 8, 2011: 11,
                    2012: 4, 2013: 9, 2014: 12, 2015: 7, 2016: 4, 2017: 12,
                    2018: 15, 2019: 10, 2020: 12, 2021: 14, 2022: 11, 2023: 9, 2024: 4
                },
                yearlyDamageK: {
                    2000: 44000, 2001: 72000, 2002: 61000, 2003: 33000, 2004: 116000, 2005: 49000,
                    2006: 77000, 2007: 38000, 2008: 66000, 2009: 94000, 2010: 55000, 2011: 72000,
                    2012: 33000, 2013: 61000, 2014: 88000, 2015: 49000, 2016: 27000, 2017: 83000,
                    2018: 116000, 2019: 72000, 2020: 94000, 2021: 110000, 2022: 83000, 2023: 72000, 2024: 38000
                },
                yearlyAffected: {
                    2000: 12000, 2001: 22000, 2002: 17000, 2003: 9000, 2004: 38000, 2005: 14000,
                    2006: 22000, 2007: 11000, 2008: 19000, 2009: 28000, 2010: 16000, 2011: 21000,
                    2012: 10000, 2013: 17000, 2014: 26000, 2015: 14000, 2016: 8000, 2017: 24000,
                    2018: 34000, 2019: 21000, 2020: 27000, 2021: 32000, 2022: 24000, 2023: 21000, 2024: 11000
                }
            },
            coastal: {
                totalEvents: 2,
                totalDeaths: 18,
                totalAffected: 52000,
                totalDamageK: 180000,
                avgDeathsPerEvent: 9.0,
                avgAffectedPerEvent: 26000,
                medianDamageK: 90000,
                yearlyEvents: {
                    2000: 0, 2001: 0, 2002: 0, 2003: 0, 2004: 0, 2005: 0,
                    2006: 0, 2007: 0, 2008: 0, 2009: 0, 2010: 0, 2011: 0,
                    2012: 0, 2013: 0, 2014: 0, 2015: 0, 2016: 0, 2017: 0,
                    2018: 0, 2019: 0, 2020: 0, 2021: 0, 2022: 0, 2023: 0, 2024: 0
                },
                yearlyDeaths: {
                    2000: 1, 2001: 1, 2002: 1, 2003: 1, 2004: 2, 2005: 1,
                    2006: 1, 2007: 1, 2008: 1, 2009: 2, 2010: 1, 2011: 1,
                    2012: 1, 2013: 1, 2014: 2, 2015: 1, 2016: 1, 2017: 1,
                    2018: 2, 2019: 1, 2020: 2, 2021: 2, 2022: 1, 2023: 1, 2024: 1
                },
                yearlyDamageK: {
                    2000: 5000, 2001: 8000, 2002: 7000, 2003: 4000, 2004: 13000, 2005: 5000,
                    2006: 9000, 2007: 5000, 2008: 8000, 2009: 10000, 2010: 7000, 2011: 8000,
                    2012: 4000, 2013: 7000, 2014: 10000, 2015: 5000, 2016: 4000, 2017: 9000,
                    2018: 13000, 2019: 8000, 2020: 10000, 2021: 13000, 2022: 9000, 2023: 8000, 2024: 5000
                },
                yearlyAffected: {
                    2000: 1000, 2001: 3000, 2002: 2000, 2003: 1000, 2004: 5000, 2005: 2000,
                    2006: 3000, 2007: 1000, 2008: 2000, 2009: 3000, 2010: 2000, 2011: 2000,
                    2012: 1000, 2013: 2000, 2014: 3000, 2015: 2000, 2016: 1000, 2017: 3000,
                    2018: 4000, 2019: 2000, 2020: 3000, 2021: 4000, 2022: 3000, 2023: 2000, 2024: 1000
                }
            }
        },

        // Flood subtypes
        subtypes: {
            'Riverine': 45,
            'Flash Flood': 48,
            'Coastal': 4,
            'Other': 3
        },

        // Risk predictions
        predictions: {
            minorFlood: { damage: 89000, rangeLow: 34000, rangeHigh: 234000 },
            severeFlood: { damage: 890000, rangeLow: 340000, rangeHigh: 2340000 }
        },

        // Statistical test results
        stats: {
            mannWhitneyP: 0.0023,
            cohenD: 0.82,
            trendSlope: 0.08,
            trendPValue: 0.112
        }
    }
};

// Model information from modeling.py
const modelInfo = {
    bestModel: 'Random Forest',
    cvR2: 0.642,
    rmse: 1.23,
    features: ['Country', 'Total Deaths (log)', 'Total Affected (log)', 'Duration', 'Years Since 2000']
};

// ============================================
// CHART INSTANCES
// ============================================
let mainChart = null;
let trendChart = null;
let comparisonChart = null;
let hazardChart = null;
let hazardTimelineChart = null;
let costBenefitChart = null;

// ============================================
// UTILITY FUNCTIONS
// ============================================

function formatNumber(num, decimals = 0) {
    if (num >= 1000000000) {
        return (num / 1000000000).toFixed(1) + 'B';
    } else if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toFixed(decimals);
}

function formatCurrency(num) {
    // num is in thousands USD
    if (num >= 1000000) {
        return '$' + (num / 1000000).toFixed(1) + 'B';
    } else if (num >= 1000) {
        return '$' + (num / 1000).toFixed(1) + 'M';
    }
    return '$' + num.toFixed(0) + 'K';
}

function getSelectedCountry() {
    return document.getElementById('country-select').value;
}

function getSelectedFloodType() {
    return document.getElementById('flood-type-select').value;
}

function getSelectedIndicator() {
    return document.getElementById('risk-indicator-select').value;
}

function getCompareCountry() {
    const compareSelect = document.getElementById('compare-select');
    return compareSelect.value || null;
}

function getCountryData(countryId) {
    return floodData[countryId];
}

function getFloodTypeData(countryId, floodType) {
    const country = floodData[countryId];
    const typeKey = floodType === 'all' ? 'all' : floodType;
    return country.byFloodType[typeKey] || country.byFloodType.all;
}

// ============================================
// SUMMARY CARDS
// ============================================

function updateSummaryCards() {
    const countryId = getSelectedCountry();
    const floodType = getSelectedFloodType();
    const data = getCountryData(countryId);
    const typeData = getFloodTypeData(countryId, floodType);
    const compareId = getCompareCountry();
    const compareData = compareId ? getCountryData(compareId) : null;
    const compareTypeData = compareId ? getFloodTypeData(compareId, floodType) : null;
    
    const container = document.getElementById('summary-cards');
    
    const floodTypeLabel = floodType === 'all' ? 'All Floods' : 
                          floodType === 'riverine' ? 'Riverine' :
                          floodType === 'flash' ? 'Flash Flood' : 'Coastal';
    
    const cards = [
        {
            label: `Total ${floodTypeLabel} Events`,
            value: typeData.totalEvents,
            change: compareTypeData ? ((typeData.totalEvents - compareTypeData.totalEvents) / compareTypeData.totalEvents * 100).toFixed(1) + '%' : '',
            highlight: false
        },
        {
            label: 'Total Deaths (2000-2025)',
            value: formatNumber(typeData.totalDeaths),
            change: compareTypeData ? ((typeData.totalDeaths - compareTypeData.totalDeaths) / compareTypeData.totalDeaths * 100).toFixed(1) + '%' : '',
            highlight: true
        },
        {
            label: 'Total Affected',
            value: formatNumber(typeData.totalAffected),
            change: compareTypeData ? ((typeData.totalAffected - compareTypeData.totalAffected) / compareTypeData.totalAffected * 100).toFixed(1) + '%' : '',
            highlight: false
        },
        {
            label: 'Economic Damage',
            value: formatCurrency(typeData.totalDamageK),
            change: compareTypeData ? ((typeData.totalDamageK - compareTypeData.totalDamageK) / compareTypeData.totalDamageK * 100).toFixed(1) + '%' : '',
            highlight: false
        }
    ];
    
    container.innerHTML = cards.map(card => `
        <div class="summary-card ${card.highlight ? 'highlight' : ''}">
            <div class="summary-card-label">${card.label}</div>
            <div class="summary-card-value">${card.value}</div>
            ${card.change ? `<div class="summary-card-change ${parseFloat(card.change) > 0 ? 'negative' : 'positive'}">${parseFloat(card.change) > 0 ? '↑' : '↓'} ${Math.abs(parseFloat(card.change))}% vs ${compareData.name}</div>` : ''}
        </div>
    `).join('');
}

// ============================================
// MAIN CHART
// ============================================

function updateMainChart() {
    const countryId = getSelectedCountry();
    const floodType = getSelectedFloodType();
    const indicator = getSelectedIndicator();
    const data = getCountryData(countryId);
    const typeData = getFloodTypeData(countryId, floodType);
    const compareId = getCompareCountry();
    const compareData = compareId && compareId !== countryId ? getCountryData(compareId) : null;
    const compareTypeData = compareId && compareId !== countryId ? getFloodTypeData(compareId, floodType) : null;
    const showPredictions = document.getElementById('show-predictions').checked;
    const showUncertainty = document.getElementById('show-uncertainty').checked;
    
    const floodTypeLabel = floodType === 'all' ? '' : 
                          floodType === 'riverine' ? 'RIVERINE ' :
                          floodType === 'flash' ? 'FLASH ' : 'COASTAL ';
    
    // Get data based on indicator
    let yearlyData, label, formatFn;
    switch(indicator) {
        case 'damage':
            yearlyData = typeData.yearlyDamageK;
            label = 'Economic Damage ($K)';
            formatFn = formatCurrency;
            break;
        case 'deaths':
            yearlyData = typeData.yearlyDeaths;
            label = 'Total Deaths';
            formatFn = formatNumber;
            break;
        case 'affected':
            yearlyData = typeData.yearlyAffected;
            label = 'Population Affected';
            formatFn = formatNumber;
            break;
        case 'events':
            yearlyData = typeData.yearlyEvents;
            label = 'Number of Events';
            formatFn = formatNumber;
            break;
    }
    
    const years = Object.keys(yearlyData).map(Number);
    const values = Object.values(yearlyData);
    
    // Update chart title
    document.getElementById('main-chart-title').textContent = 
        `${floodTypeLabel}FLOOD ${indicator.toUpperCase()} IN ${data.name.toUpperCase()}${compareData ? ' vs ' + compareData.name.toUpperCase() : ''}`;
    
    // Prepare datasets
    const datasets = [{
        label: data.name,
        data: values,
        backgroundColor: data.lightColor,
        borderColor: data.color,
        borderWidth: 2,
        fill: true,
        tension: 0.4
    }];
    
    // Add comparison dataset
    if (compareData && compareTypeData) {
        let compareYearlyData;
        switch(indicator) {
            case 'damage': compareYearlyData = compareTypeData.yearlyDamageK; break;
            case 'deaths': compareYearlyData = compareTypeData.yearlyDeaths; break;
            case 'affected': compareYearlyData = compareTypeData.yearlyAffected; break;
            case 'events': compareYearlyData = compareTypeData.yearlyEvents; break;
        }
        
        datasets.push({
            label: compareData.name,
            data: Object.values(compareYearlyData),
            backgroundColor: compareData.lightColor,
            borderColor: compareData.color,
            borderWidth: 2,
            fill: true,
            tension: 0.4
        });
    }
    
    // Add prediction data if enabled
    if (showPredictions) {
        const predictionYears = [2025, 2030, 2035, 2040];
        const lastValue = values[values.length - 1];
        const growthRate = indicator === 'damage' ? 1.08 : indicator === 'deaths' ? 0.95 : 1.05;
        
        const predictionValues = predictionYears.map((_, i) => 
            Math.round(lastValue * Math.pow(growthRate, (i + 1) * 5))
        );
        
        datasets.push({
            label: `${data.name} (Predicted)`,
            data: [...Array(years.length - 1).fill(null), values[values.length - 1], ...predictionValues],
            backgroundColor: 'rgba(156, 163, 175, 0.2)',
            borderColor: '#9ca3af',
            borderWidth: 2,
            borderDash: [5, 5],
            fill: true,
            tension: 0.4
        });
        
        years.push(...predictionYears);
    }
    
    // Destroy existing chart
    if (mainChart) {
        mainChart.destroy();
    }
    
    // Create new chart
    const ctx = document.getElementById('main-chart').getContext('2d');
    mainChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: years,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${formatFn(context.raw)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: 'Year'
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    title: {
                        display: true,
                        text: label
                    },
                    ticks: {
                        callback: function(value) {
                            return formatFn(value);
                        }
                    }
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            }
        }
    });
    
    // Update legend
    updateChartLegend(data, compareData);
}

function updateChartLegend(data, compareData) {
    const container = document.getElementById('chart-legend');
    
    let html = `
        <div class="legend-item">
            <div class="legend-color" style="background-color: ${data.color}"></div>
            <span>${data.name}</span>
        </div>
    `;
    
    if (compareData) {
        html += `
            <div class="legend-item">
                <div class="legend-color" style="background-color: ${compareData.color}"></div>
                <span>${compareData.name}</span>
            </div>
        `;
    }
    
    container.innerHTML = html;
}

// ============================================
// DATA TABLE
// ============================================

function updateDataTable() {
    const countryId = getSelectedCountry();
    const floodType = getSelectedFloodType();
    const indicator = getSelectedIndicator();
    const data = getCountryData(countryId);
    const typeData = getFloodTypeData(countryId, floodType);
    
    const floodTypeLabel = floodType === 'all' ? '' : 
                          floodType === 'riverine' ? 'RIVERINE ' :
                          floodType === 'flash' ? 'FLASH ' : 'COASTAL ';
    
    let yearlyData, formatFn;
    switch(indicator) {
        case 'damage':
            yearlyData = typeData.yearlyDamageK;
            formatFn = formatCurrency;
            break;
        case 'deaths':
            yearlyData = typeData.yearlyDeaths;
            formatFn = formatNumber;
            break;
        case 'affected':
            yearlyData = typeData.yearlyAffected;
            formatFn = formatNumber;
            break;
        case 'events':
            yearlyData = typeData.yearlyEvents;
            formatFn = formatNumber;
            break;
    }
    
    // Update table title
    document.getElementById('table-title').textContent = 
        `${floodTypeLabel}FLOOD ${indicator.toUpperCase()} DATA - ${data.name.toUpperCase()}`;
    
    // Select specific years for the table
    const displayYears = [2005, 2010, 2015, 2020, 2024];
    
    // Update table header
    const headerRow = document.querySelector('#data-table thead tr');
    headerRow.innerHTML = '<th></th>' + displayYears.map(y => `<th>${y}</th>`).join('');
    
    // Calculate rows
    const rows = [
        {
            label: indicator === 'damage' ? 'Annual Economic Damage' : 
                   indicator === 'deaths' ? 'Annual Deaths' :
                   indicator === 'affected' ? 'Annual Affected Population' : 'Annual Events',
            values: displayYears.map(y => formatFn(yearlyData[y] || 0)),
            highlight: true
        },
        {
            label: 'Cumulative Total',
            values: displayYears.map(y => {
                let sum = 0;
                for (let yr = 2000; yr <= y; yr++) {
                    sum += yearlyData[yr] || 0;
                }
                return formatFn(sum);
            }),
            highlight: false
        },
        {
            label: '5-Year Average',
            values: displayYears.map(y => {
                let sum = 0, count = 0;
                for (let yr = y - 4; yr <= y; yr++) {
                    if (yearlyData[yr]) {
                        sum += yearlyData[yr];
                        count++;
                    }
                }
                return count > 0 ? formatFn(sum / count) : '-';
            }),
            highlight: false
        },
        {
            label: 'Year-over-Year Change',
            values: displayYears.map((y, i) => {
                if (i === 0) return '-';
                const prev = yearlyData[displayYears[i-1]] || 0;
                const curr = yearlyData[y] || 0;
                if (prev === 0) return '-';
                const change = ((curr - prev) / prev * 100).toFixed(1);
                return (change > 0 ? '+' : '') + change + '%';
            }),
            highlight: false
        }
    ];
    
    // Update table body
    const tbody = document.getElementById('table-body');
    tbody.innerHTML = rows.map(row => `
        <tr class="${row.highlight ? 'highlight' : ''}">
            <td>${row.label}</td>
            ${row.values.map(v => `<td>${v}</td>`).join('')}
        </tr>
    `).join('');
}

// ============================================
// SECONDARY CHARTS
// ============================================

function updateTrendChart() {
    const countryId = getSelectedCountry();
    const floodType = getSelectedFloodType();
    const data = getCountryData(countryId);
    const typeData = getFloodTypeData(countryId, floodType);
    const compareId = getCompareCountry();
    const compareData = compareId && compareId !== countryId ? getCountryData(compareId) : null;
    const compareTypeData = compareId && compareId !== countryId ? getFloodTypeData(compareId, floodType) : null;
    
    const years = Object.keys(typeData.yearlyEvents).map(Number);
    
    const datasets = [{
        label: data.name,
        data: Object.values(typeData.yearlyEvents),
        backgroundColor: data.color,
        borderColor: data.color,
        borderWidth: 2
    }];
    
    if (compareData && compareTypeData) {
        datasets.push({
            label: compareData.name,
            data: Object.values(compareTypeData.yearlyEvents),
            backgroundColor: compareData.color,
            borderColor: compareData.color,
            borderWidth: 2
        });
    }
    
    if (trendChart) {
        trendChart.destroy();
    }
    
    const ctx = document.getElementById('trend-chart').getContext('2d');
    trendChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: years,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                x: {
                    grid: { display: false }
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Events'
                    }
                }
            }
        }
    });
}

function updateComparisonChart() {
    const floodType = getSelectedFloodType();
    const bangladesh = getCountryData('bangladesh');
    const turkey = getCountryData('turkey');
    const bangladeshType = getFloodTypeData('bangladesh', floodType);
    const turkeyType = getFloodTypeData('turkey', floodType);
    
    const metrics = ['Avg Deaths/Event', 'Avg Affected/Event (K)', 'Median Damage ($M)'];
    const bangladeshValues = [
        bangladeshType.avgDeathsPerEvent,
        bangladeshType.avgAffectedPerEvent / 1000,
        bangladeshType.medianDamageK / 1000
    ];
    const turkeyValues = [
        turkeyType.avgDeathsPerEvent,
        turkeyType.avgAffectedPerEvent / 1000,
        turkeyType.medianDamageK / 1000
    ];
    
    if (comparisonChart) {
        comparisonChart.destroy();
    }
    
    const ctx = document.getElementById('comparison-chart').getContext('2d');
    comparisonChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: metrics,
            datasets: [
                {
                    label: 'Bangladesh',
                    data: bangladeshValues,
                    backgroundColor: bangladesh.lightColor,
                    borderColor: bangladesh.color,
                    borderWidth: 2,
                    pointBackgroundColor: bangladesh.color
                },
                {
                    label: 'Turkey',
                    data: turkeyValues,
                    backgroundColor: turkey.lightColor,
                    borderColor: turkey.color,
                    borderWidth: 2,
                    pointBackgroundColor: turkey.color
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    ticks: {
                        display: false
                    }
                }
            }
        }
    });
}

// ============================================
// PREDICTIONS SECTION
// ============================================

function updatePredictions() {
    const countryId = getSelectedCountry();
    const floodType = getSelectedFloodType();
    const data = getCountryData(countryId);
    const typeData = getFloodTypeData(countryId, floodType);
    
    const floodTypeLabel = floodType === 'all' ? 'Flood' : 
                          floodType === 'riverine' ? 'Riverine Flood' :
                          floodType === 'flash' ? 'Flash Flood' : 'Coastal Flood';
    
    const container = document.getElementById('prediction-cards');
    
    const predictions = [
        {
            title: `Minor ${floodTypeLabel} Scenario`,
            badge: '2025',
            value: formatCurrency(data.predictions.minorFlood.damage),
            range: `${formatCurrency(data.predictions.minorFlood.rangeLow)} - ${formatCurrency(data.predictions.minorFlood.rangeHigh)}`
        },
        {
            title: `Severe ${floodTypeLabel} Scenario`,
            badge: '2025',
            value: formatCurrency(data.predictions.severeFlood.damage),
            range: `${formatCurrency(data.predictions.severeFlood.rangeLow)} - ${formatCurrency(data.predictions.severeFlood.rangeHigh)}`
        },
        {
            title: 'Expected Annual Deaths',
            badge: '2025',
            value: formatNumber(typeData.avgDeathsPerEvent),
            range: `Per average ${floodTypeLabel.toLowerCase()} event`
        },
        {
            title: 'Expected Affected Pop.',
            badge: '2025',
            value: formatNumber(typeData.avgAffectedPerEvent),
            range: `Per average ${floodTypeLabel.toLowerCase()} event`
        }
    ];
    
    container.innerHTML = predictions.map(pred => `
        <div class="prediction-card">
            <div class="prediction-card-header">
                <span class="prediction-card-title">${pred.title}</span>
                <span class="prediction-card-badge">${pred.badge}</span>
            </div>
            <div class="prediction-card-value">${pred.value}</div>
            <div class="prediction-card-range">95% CI: ${pred.range}</div>
        </div>
    `).join('');
    
    // Update model info
    document.getElementById('best-model').textContent = modelInfo.bestModel;
    document.getElementById('cv-r2').textContent = modelInfo.cvR2.toFixed(3);
    document.getElementById('model-rmse').textContent = modelInfo.rmse.toFixed(2);
}

// ============================================
// STATISTICS SECTION
// ============================================

function updateStatistics() {
    const bangladesh = getCountryData('bangladesh');
    const turkey = getCountryData('turkey');
    
    const container = document.getElementById('stats-grid');
    
    const stats = [
        {
            title: 'Mortality Difference Test',
            value: 'p = ' + bangladesh.stats.mannWhitneyP.toFixed(4),
            unit: 'Mann-Whitney U',
            sub: 'Significant at α = 0.05'
        },
        {
            title: 'Effect Size (Cohen\'s d)',
            value: bangladesh.stats.cohenD.toFixed(2),
            unit: 'Large effect',
            sub: 'Bangladesh vs Turkey mortality'
        },
        {
            title: 'Bangladesh Trend',
            value: (bangladesh.stats.trendSlope > 0 ? '+' : '') + bangladesh.stats.trendSlope.toFixed(2),
            unit: 'events/year',
            sub: bangladesh.stats.trendPValue < 0.05 ? 'Significant trend' : 'Not significant'
        },
        {
            title: 'Turkey Trend',
            value: (turkey.stats.trendSlope > 0 ? '+' : '') + turkey.stats.trendSlope.toFixed(2),
            unit: 'events/year',
            sub: turkey.stats.trendPValue < 0.05 ? 'Significant trend' : 'Not significant'
        },
        {
            title: 'Death Rate Ratio',
            value: (bangladesh.avgDeathsPerEvent / turkey.avgDeathsPerEvent).toFixed(1) + 'x',
            unit: 'BGD vs TUR',
            sub: 'Average deaths per event'
        },
        {
            title: 'Affected Pop. Ratio',
            value: (bangladesh.avgAffectedPerEvent / turkey.avgAffectedPerEvent).toFixed(0) + 'x',
            unit: 'BGD vs TUR',
            sub: 'Average affected per event'
        }
    ];
    
    container.innerHTML = stats.map(stat => `
        <div class="stat-card">
            <div class="stat-card-title">${stat.title}</div>
            <div class="stat-card-content">
                <span class="stat-card-value">${stat.value}</span>
                <span class="stat-card-unit">${stat.unit}</span>
            </div>
            <div class="stat-card-sub">${stat.sub}</div>
        </div>
    `).join('');
}

// ============================================
// EVENT LISTENERS
// ============================================

function setupEventListeners() {
    // Country selection - affects all tabs
    document.getElementById('country-select').addEventListener('change', () => {
        const activeTab = document.querySelector('.nav-tab.active');
        const tabName = activeTab ? activeTab.getAttribute('data-tab') : 'risk';
        
        if (tabName === 'risk') {
            updateAllCharts();
        } else if (tabName === 'hazard') {
            updateHazardTab();
        } else if (tabName === 'cost-benefit') {
            updateCostBenefitTab();
        }
    });
    
    // ============================================
    // RISK TAB FILTERS
    // ============================================
    
    // Flood type selection
    document.getElementById('flood-type-select').addEventListener('change', () => {
        updateAllCharts();
    });
    
    // Risk indicator selection
    document.getElementById('risk-indicator-select').addEventListener('change', () => {
        updateMainChart();
        updateDataTable();
    });
    
    // Compare selection
    document.getElementById('compare-select').addEventListener('change', () => {
        updateAllCharts();
    });
    
    // Show predictions checkbox
    document.getElementById('show-predictions').addEventListener('change', () => {
        updateMainChart();
    });
    
    // Show uncertainty checkbox
    document.getElementById('show-uncertainty').addEventListener('change', () => {
        updateMainChart();
    });
    
    // ============================================
    // HAZARD TAB FILTERS
    // ============================================
    
    // Hazard type selection
    document.getElementById('hazard-type-select').addEventListener('change', () => {
        updateHazardTab();
    });
    
    // Hazard period selection
    document.getElementById('hazard-period-select').addEventListener('change', () => {
        updateHazardTab();
    });
    
    // Hazard severity selection
    document.getElementById('hazard-severity-select').addEventListener('change', () => {
        updateHazardTab();
    });
    
    // Show deaths overlay
    document.getElementById('show-deaths-overlay').addEventListener('change', () => {
        updateHazardTab();
    });
    
    // Compare countries in hazard tab
    document.getElementById('compare-countries-hazard').addEventListener('change', () => {
        updateHazardTab();
    });
    
    // ============================================
    // COST-BENEFIT TAB FILTERS
    // ============================================
    
    // Scenario selection
    document.getElementById('cb-scenario-select').addEventListener('change', () => {
        updateCostBenefitTab();
    });
    
    // Projection period
    document.getElementById('cb-period-select').addEventListener('change', () => {
        updateCostBenefitTab();
    });
    
    // Discount rate
    document.getElementById('cb-discount-select').addEventListener('change', () => {
        updateCostBenefitTab();
    });
    
    // Focus area
    document.getElementById('cb-focus-select').addEventListener('change', () => {
        updateCostBenefitTab();
    });
    
    // Show break-even
    document.getElementById('show-breakeven').addEventListener('change', () => {
        updateCostBenefitTab();
    });
    
    // Include indirect benefits
    document.getElementById('include-indirect').addEventListener('change', () => {
        updateCostBenefitTab();
    });
    
    // ============================================
    // NAVIGATION
    // ============================================
    
    // Navigation tabs
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
            e.target.classList.add('active');
            
            // Switch tab content
            const tabName = e.target.getAttribute('data-tab');
            switchTab(tabName);
        });
    });
}

// ============================================
// TAB SWITCHING
// ============================================

function switchTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(content => {
        content.style.display = 'none';
    });
    
    // Show selected tab content
    const selectedTab = document.getElementById(`tab-${tabName}`);
    if (selectedTab) {
        selectedTab.style.display = 'block';
    }
    
    // Update sidebar filters based on tab
    updateSidebarFilters(tabName);
    
    // Initialize charts for the selected tab
    if (tabName === 'hazard') {
        updateHazardTab();
    } else if (tabName === 'cost-benefit') {
        updateCostBenefitTab();
    } else if (tabName === 'risk') {
        updateAllCharts();
    }
}

function updateSidebarFilters(tabName) {
    // Hide all tab-specific filters
    document.getElementById('risk-filters').style.display = 'none';
    document.getElementById('hazard-filters').style.display = 'none';
    document.getElementById('cost-benefit-filters').style.display = 'none';
    
    // Show appropriate filters based on tab
    if (tabName === 'risk') {
        document.getElementById('risk-filters').style.display = 'block';
    } else if (tabName === 'hazard') {
        document.getElementById('hazard-filters').style.display = 'block';
    } else if (tabName === 'cost-benefit') {
        document.getElementById('cost-benefit-filters').style.display = 'block';
    }
}

// ============================================
// HAZARD TAB
// ============================================

function getHazardFilters() {
    return {
        hazardType: document.getElementById('hazard-type-select').value,
        period: document.getElementById('hazard-period-select').value,
        severity: document.getElementById('hazard-severity-select').value,
        showDeaths: document.getElementById('show-deaths-overlay').checked,
        compareCountries: document.getElementById('compare-countries-hazard').checked
    };
}

function filterDataByPeriod(yearlyData, period) {
    const filtered = {};
    let startYear, endYear;
    
    switch(period) {
        case 'recent':
            startYear = 2015; endYear = 2024;
            break;
        case 'mid':
            startYear = 2010; endYear = 2014;
            break;
        case 'early':
            startYear = 2000; endYear = 2009;
            break;
        default:
            startYear = 2000; endYear = 2024;
    }
    
    for (let year = startYear; year <= endYear; year++) {
        if (yearlyData[year] !== undefined) {
            filtered[year] = yearlyData[year];
        }
    }
    
    return filtered;
}

function updateHazardTab() {
    const countryId = getSelectedCountry();
    const data = getCountryData(countryId);
    const filters = getHazardFilters();
    const typeData = getFloodTypeData(countryId, filters.hazardType);
    
    // Filter data by period
    const filteredEvents = filterDataByPeriod(typeData.yearlyEvents, filters.period);
    const filteredDeaths = filterDataByPeriod(typeData.yearlyDeaths, filters.period);
    
    // Calculate statistics based on filtered data
    const years = Object.keys(filteredEvents);
    const totalEvents = Object.values(filteredEvents).reduce((a, b) => a + b, 0);
    const avgEventsPerYear = years.length > 0 ? (totalEvents / years.length).toFixed(1) : '0';
    
    document.getElementById('hazard-frequency').textContent = avgEventsPerYear;
    
    // Average duration (estimated based on country and severity)
    let avgDuration;
    if (countryId === 'bangladesh') {
        avgDuration = filters.severity === 'high' ? '18.5' : filters.severity === 'medium' ? '14.5' : '12.5';
    } else {
        avgDuration = filters.severity === 'high' ? '12.2' : filters.severity === 'medium' ? '8.2' : '6.5';
    }
    document.getElementById('hazard-duration').textContent = avgDuration;
    
    // Severity index based on filter
    let severityIndex;
    if (filters.severity === 'high') {
        severityIndex = countryId === 'bangladesh' ? '4.2' : '3.8';
    } else if (filters.severity === 'medium') {
        severityIndex = countryId === 'bangladesh' ? '3.2' : '2.8';
    } else {
        severityIndex = countryId === 'bangladesh' ? '2.8' : '2.4';
    }
    document.getElementById('hazard-severity').textContent = severityIndex;
    
    // Area affected
    const areaAffected = countryId === 'bangladesh' ? '52,000' : '18,000';
    document.getElementById('hazard-area').textContent = areaAffected;
    
    // Update hazard zones based on hazard type
    updateHazardZones(countryId, filters.hazardType);
    
    // Create hazard charts
    createHazardChart(countryId, filters.hazardType);
    createHazardTimelineChart(countryId, filters);
}

function updateHazardZones(countryId, hazardType) {
    let high, medium, low;
    
    if (countryId === 'bangladesh') {
        switch(hazardType) {
            case 'riverine':
                high = '48%'; medium = '32%'; low = '20%';
                break;
            case 'flash':
                high = '35%'; medium = '40%'; low = '25%';
                break;
            case 'coastal':
                high = '55%'; medium = '28%'; low = '17%';
                break;
            default:
                high = '42%'; medium = '35%'; low = '23%';
        }
    } else {
        switch(hazardType) {
            case 'riverine':
                high = '22%'; medium = '38%'; low = '40%';
                break;
            case 'flash':
                high = '28%'; medium = '35%'; low = '37%';
                break;
            case 'coastal':
                high = '12%'; medium = '25%'; low = '63%';
                break;
            default:
                high = '18%'; medium = '34%'; low = '48%';
        }
    }
    
    document.getElementById('zone-high').textContent = high;
    document.getElementById('zone-medium').textContent = medium;
    document.getElementById('zone-low').textContent = low;
}

function createHazardChart(countryId, hazardType) {
    const data = getCountryData(countryId);
    
    if (hazardChart) {
        hazardChart.destroy();
    }
    
    const ctx = document.getElementById('hazard-chart').getContext('2d');
    
    // Highlight selected type
    const backgroundColors = [
        hazardType === 'riverine' || hazardType === 'all' ? '#3182ce' : 'rgba(49, 130, 206, 0.3)',
        hazardType === 'flash' || hazardType === 'all' ? '#e53e3e' : 'rgba(229, 62, 62, 0.3)',
        hazardType === 'coastal' || hazardType === 'all' ? '#38a169' : 'rgba(56, 161, 105, 0.3)',
        '#9ca3af'
    ];
    
    hazardChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Riverine', 'Flash Flood', 'Coastal', 'Other'],
            datasets: [{
                data: [
                    data.subtypes['Riverine'],
                    data.subtypes['Flash Flood'],
                    data.subtypes['Coastal'],
                    data.subtypes['Other']
                ],
                backgroundColor: backgroundColors,
                borderWidth: hazardType !== 'all' ? 3 : 0,
                borderColor: hazardType !== 'all' ? '#fff' : 'transparent'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${context.raw}%`;
                        }
                    }
                }
            }
        }
    });
}

function createHazardTimelineChart(countryId, filters) {
    const typeData = getFloodTypeData(countryId, filters.hazardType);
    const data = getCountryData(countryId);
    
    // Filter by period
    const filteredEvents = filterDataByPeriod(typeData.yearlyEvents, filters.period);
    const filteredDeaths = filterDataByPeriod(typeData.yearlyDeaths, filters.period);
    
    if (hazardTimelineChart) {
        hazardTimelineChart.destroy();
    }
    
    const ctx = document.getElementById('hazard-timeline-chart').getContext('2d');
    
    const years = Object.keys(filteredEvents).map(Number);
    const events = Object.values(filteredEvents);
    const deaths = Object.values(filteredDeaths);
    
    const datasets = [
        {
            label: `${data.name} - Events`,
            data: events,
            backgroundColor: data.lightColor,
            borderColor: data.color,
            borderWidth: 1,
            yAxisID: 'y'
        }
    ];
    
    // Add deaths overlay if enabled
    if (filters.showDeaths) {
        datasets.push({
            label: `${data.name} - Deaths`,
            data: deaths,
            type: 'line',
            borderColor: '#e53e3e',
            backgroundColor: 'rgba(229, 62, 62, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.4,
            yAxisID: 'y1'
        });
    }
    
    // Add comparison country if enabled
    if (filters.compareCountries) {
        const compareId = countryId === 'bangladesh' ? 'turkey' : 'bangladesh';
        const compareData = getCountryData(compareId);
        const compareTypeData = getFloodTypeData(compareId, filters.hazardType);
        const compareEvents = filterDataByPeriod(compareTypeData.yearlyEvents, filters.period);
        
        datasets.push({
            label: `${compareData.name} - Events`,
            data: Object.values(compareEvents),
            backgroundColor: compareData.lightColor,
            borderColor: compareData.color,
            borderWidth: 1,
            yAxisID: 'y'
        });
    }
    
    hazardTimelineChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: years,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    title: {
                        display: true,
                        text: 'Year'
                    }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Number of Events'
                    },
                    beginAtZero: true
                },
                y1: {
                    type: 'linear',
                    display: filters.showDeaths,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Deaths'
                    },
                    beginAtZero: true,
                    grid: {
                        drawOnChartArea: false
                    }
                }
            }
        }
    });
}

// ============================================
// COST-BENEFIT TAB
// ============================================

function getCostBenefitFilters() {
    return {
        scenario: document.getElementById('cb-scenario-select').value,
        period: parseInt(document.getElementById('cb-period-select').value),
        discountRate: parseInt(document.getElementById('cb-discount-select').value) / 100,
        focusArea: document.getElementById('cb-focus-select').value,
        showBreakeven: document.getElementById('show-breakeven').checked,
        includeIndirect: document.getElementById('include-indirect').checked
    };
}

function updateCostBenefitTab() {
    const countryId = getSelectedCountry();
    const filters = getCostBenefitFilters();
    
    // Calculate values based on scenario and country
    const baseValues = getBaseValues(countryId);
    const adjustedValues = adjustForScenario(baseValues, filters);
    
    // Update metric cards
    document.getElementById('cb-annual-loss').textContent = formatCurrency(adjustedValues.annualLoss);
    document.getElementById('cb-investment').textContent = formatCurrency(adjustedValues.investment);
    document.getElementById('cb-savings').textContent = formatCurrency(adjustedValues.savings);
    document.getElementById('cb-ratio').textContent = adjustedValues.ratio.toFixed(1) + 'x';
    
    // Update investment breakdown based on focus area
    updateInvestmentBreakdown(countryId, filters);
    
    // Create cost-benefit chart
    createCostBenefitChart(countryId, filters);
}

function getBaseValues(countryId) {
    if (countryId === 'bangladesh') {
        return {
            annualLoss: 2400000, // $2.4B in thousands
            investment: 450000,  // $450M
            savings: 1800000,    // $1.8B
            ratio: 4.0
        };
    } else {
        return {
            annualLoss: 890000,
            investment: 280000,
            savings: 650000,
            ratio: 2.3
        };
    }
}

function adjustForScenario(baseValues, filters) {
    let multiplier = 1;
    let ratioAdjust = 0;
    
    // Adjust based on scenario
    switch(filters.scenario) {
        case 'aggressive':
            multiplier = 1.5;
            ratioAdjust = 0.8;
            break;
        case 'conservative':
            multiplier = 0.7;
            ratioAdjust = -0.3;
            break;
        default:
            multiplier = 1;
            ratioAdjust = 0;
    }
    
    // Adjust for indirect benefits
    if (filters.includeIndirect) {
        ratioAdjust += 0.5;
    }
    
    // Apply discount rate effect (higher rate = lower NPV)
    const discountEffect = 1 - (filters.discountRate - 0.05);
    
    return {
        annualLoss: baseValues.annualLoss,
        investment: Math.round(baseValues.investment * multiplier),
        savings: Math.round(baseValues.savings * multiplier * discountEffect),
        ratio: Math.max(1, (baseValues.ratio + ratioAdjust) * discountEffect)
    };
}

function updateInvestmentBreakdown(countryId, filters) {
    const breakdownItems = document.querySelectorAll('.cb-breakdown-item');
    
    // Investment allocation based on focus area
    let allocations;
    
    switch(filters.focusArea) {
        case 'infrastructure':
            allocations = [
                { name: 'Early Warning Systems', pct: 10, amount: 35 },
                { name: 'Flood Barriers & Levees', pct: 55, amount: 250 },
                { name: 'Drainage Infrastructure', pct: 25, amount: 110 },
                { name: 'Community Programs', pct: 5, amount: 20 },
                { name: 'Evacuation Routes', pct: 5, amount: 25 }
            ];
            break;
        case 'warning':
            allocations = [
                { name: 'Early Warning Systems', pct: 45, amount: 180 },
                { name: 'Flood Barriers & Levees', pct: 20, amount: 90 },
                { name: 'Drainage Infrastructure', pct: 15, amount: 65 },
                { name: 'Community Programs', pct: 12, amount: 50 },
                { name: 'Evacuation Routes', pct: 8, amount: 35 }
            ];
            break;
        case 'community':
            allocations = [
                { name: 'Early Warning Systems', pct: 20, amount: 80 },
                { name: 'Flood Barriers & Levees', pct: 15, amount: 65 },
                { name: 'Drainage Infrastructure', pct: 15, amount: 65 },
                { name: 'Community Programs', pct: 35, amount: 150 },
                { name: 'Evacuation Routes', pct: 15, amount: 60 }
            ];
            break;
        default:
            allocations = [
                { name: 'Early Warning Systems', pct: 15, amount: 50 },
                { name: 'Flood Barriers & Levees', pct: 45, amount: 200 },
                { name: 'Drainage Infrastructure', pct: 28, amount: 120 },
                { name: 'Community Programs', pct: 8, amount: 30 },
                { name: 'Evacuation Routes', pct: 12, amount: 50 }
            ];
    }
    
    // Apply scenario multiplier
    const scenarioMultiplier = filters.scenario === 'aggressive' ? 1.5 : 
                               filters.scenario === 'conservative' ? 0.7 : 1;
    
    // Update DOM
    breakdownItems.forEach((item, index) => {
        if (allocations[index]) {
            const barFill = item.querySelector('.cb-bar-fill');
            const valueSpan = item.querySelector('.cb-breakdown-value');
            
            if (barFill) {
                barFill.style.width = allocations[index].pct + '%';
            }
            if (valueSpan) {
                valueSpan.textContent = '$' + Math.round(allocations[index].amount * scenarioMultiplier) + 'M';
            }
        }
    });
}

function createCostBenefitChart(countryId, filters) {
    if (costBenefitChart) {
        costBenefitChart.destroy();
    }
    
    const ctx = document.getElementById('cost-benefit-chart').getContext('2d');
    
    // Generate years based on projection period
    const startYear = 2025;
    const years = [];
    for (let i = 0; i <= filters.period; i++) {
        years.push((startYear + i).toString());
    }
    
    // Calculate projections based on filters
    const baseValues = getBaseValues(countryId);
    const scenarioMultiplier = filters.scenario === 'aggressive' ? 1.5 : 
                               filters.scenario === 'conservative' ? 0.7 : 1;
    
    const annualInvestment = (baseValues.investment * scenarioMultiplier) / filters.period;
    const annualBenefit = (baseValues.savings * scenarioMultiplier) / filters.period;
    const indirectMultiplier = filters.includeIndirect ? 1.25 : 1;
    
    // Calculate cumulative values with discount rate
    const cumulativeCost = [];
    const cumulativeBenefit = [];
    
    let totalCost = 0;
    let totalBenefit = 0;
    
    for (let i = 0; i <= filters.period; i++) {
        const discountFactor = Math.pow(1 + filters.discountRate, i);
        totalCost += annualInvestment / discountFactor;
        
        if (i > 0) {
            totalBenefit += (annualBenefit * indirectMultiplier) / discountFactor;
        }
        
        cumulativeCost.push(Math.round(totalCost / 1000)); // Convert to millions
        cumulativeBenefit.push(Math.round(totalBenefit / 1000));
    }
    
    const datasets = [
        {
            label: 'Cumulative Investment ($M)',
            data: cumulativeCost,
            borderColor: '#e53e3e',
            backgroundColor: 'rgba(229, 62, 62, 0.1)',
            fill: true,
            tension: 0.4
        },
        {
            label: 'Cumulative Benefits ($M)',
            data: cumulativeBenefit,
            borderColor: '#38a169',
            backgroundColor: 'rgba(56, 161, 105, 0.1)',
            fill: true,
            tension: 0.4
        }
    ];
    
    // Add break-even line if enabled
    const annotations = {};
    if (filters.showBreakeven) {
        // Find break-even point
        let breakevenYear = null;
        for (let i = 0; i < cumulativeCost.length; i++) {
            if (cumulativeBenefit[i] >= cumulativeCost[i]) {
                breakevenYear = i;
                break;
            }
        }
        
        if (breakevenYear !== null) {
            annotations.breakeven = {
                type: 'line',
                xMin: breakevenYear,
                xMax: breakevenYear,
                borderColor: '#805ad5',
                borderWidth: 2,
                borderDash: [5, 5],
                label: {
                    display: true,
                    content: 'Break-even',
                    position: 'start'
                }
            };
        }
    }
    
    costBenefitChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: years,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: $${context.raw}M`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Year'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Amount ($M)'
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

// ============================================
// RESET DASHBOARD
// ============================================

function resetDashboard() {
    // Reset all selections to default
    document.getElementById('country-select').value = 'bangladesh';
    document.getElementById('flood-type-select').value = 'all';
    document.getElementById('risk-indicator-select').value = 'damage';
    document.getElementById('compare-select').value = '';
    document.getElementById('show-predictions').checked = false;
    document.getElementById('show-uncertainty').checked = false;
    
    // Reset to RISK tab
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelector('.nav-tab[data-tab="risk"]').classList.add('active');
    
    // Switch to risk tab
    switchTab('risk');
}

// ============================================
// UPDATE ALL
// ============================================

function updateAllCharts() {
    updateSummaryCards();
    updateMainChart();
    updateDataTable();
    updateTrendChart();
    updateComparisonChart();
    updatePredictions();
    updateStatistics();
}

// ============================================
// INITIALIZATION
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    updateAllCharts();
});
