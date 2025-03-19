import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { 
  Container, 
  Box, 
  Typography, 
  Paper, 
  CircularProgress,
  Alert,
  ThemeProvider,
  createTheme,
  Button
} from '@mui/material';
import { CloudUpload, CheckCircle, PictureAsPdf } from '@mui/icons-material';
import axios from 'axios';
import { motion } from 'framer-motion';

const theme = createTheme({
  palette: {
    primary: {
      main: '#2196f3',
    },
    secondary: {
      main: '#f50057',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
    text: {
      primary: '#2c3e50',
      secondary: '#546e7a',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h3: {
      fontWeight: 600,
      color: '#1a237e',
    },
    h6: {
      fontWeight: 500,
    },
  },
  components: {
    MuiContainer: {
      styleOverrides: {
        root: {
          paddingTop: '2rem',
          paddingBottom: '2rem',
        },
      },
    },
  },
});

// Constants for box dimensions
const BOX_DIMENSIONS = {
  width: '100%',
  maxWidth: '450px',  // Increased width
  minHeight: '250px', // Reduced height
};

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setFile(file);
      setError(null);
      setResult(null);
      
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png'],
      'application/pdf': ['.pdf']
    },
    maxFiles: 1,
    maxSize: 10485760, // 10MB
    onDropRejected: (fileRejections) => {
      const error = fileRejections[0]?.errors[0];
      if (error?.code === 'file-too-large') {
        setError('File is too large. Maximum size is 10MB.');
      } else if (error?.code === 'file-invalid-type') {
        setError('Only images (JPEG, PNG) and PDF files are allowed.');
      } else {
        setError('Invalid file. Please try again.');
      }
    }
  });

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResult(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Error processing image. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <Box sx={{ 
        minHeight: '100vh',
        width: '100vw',
        backgroundColor: 'background.default',
        py: 4,
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'flex-start'
      }}>
        <Container maxWidth="xl" sx={{ width: '100%' }}>
          <Box sx={{ 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'center',
            gap: 4,
            width: '100%'
          }}>
            <Box sx={{ textAlign: 'center', mb: 2, width: '100%' }}>
              <Typography 
                variant="h3" 
                component="h1" 
                gutterBottom 
                sx={{ 
                  fontSize: { xs: '2rem', sm: '2.5rem' },
                  mb: 1
                }}
              >
                Brain Tumor Classification
              </Typography>
              <Typography 
                variant="subtitle1" 
                color="text.secondary" 
                sx={{ 
                  maxWidth: '600px', 
                  mx: 'auto',
                  fontSize: '1rem',
                  mb: 2
                }}
              >
                Upload a brain MRI image (JPEG, PNG) or PDF file to classify the type of tumor
              </Typography>
            </Box>

            <Box sx={{ 
              display: 'flex', 
              flexDirection: { xs: 'column', md: 'row' }, 
              gap: 3, 
              width: '100%',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <Box sx={{
                ...BOX_DIMENSIONS,
                transition: 'all 0.3s ease',
              }}>
                <div {...getRootProps()} style={{ width: '100%', height: '100%' }}>
                  <motion.div
                    style={{
                      boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                      padding: theme.spacing(2),
                      backgroundColor: isDragActive 
                        ? theme.palette.action.hover 
                        : theme.palette.background.paper,
                      border: `2px dashed ${isDragActive 
                        ? theme.palette.primary.main 
                        : theme.palette.grey[300]}`,
                      cursor: 'pointer',
                      borderRadius: '12px',
                      textAlign: 'center',
                      height: '100%',
                      display: 'flex',
                      flexDirection: 'column',
                      justifyContent: 'center',
                      minHeight: '250px',
                      position: 'relative',
                      overflow: 'hidden'
                    }}
                    whileHover={{ scale: 1.01 }}
                  >
                    <input {...getInputProps()} />
                    <Box sx={{ 
                      display: 'flex', 
                      flexDirection: 'column', 
                      alignItems: 'center', 
                      gap: 2,
                      height: '100%',
                      position: 'relative',
                      zIndex: 1
                    }}>
                      <Box sx={{ 
                        flex: 1,
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: 2
                      }}>
                        <motion.div
                          whileHover={{ rotate: 360 }}
                          transition={{ duration: 0.5 }}
                        >
                          <CloudUpload sx={{ 
                            fontSize: 60, 
                            color: 'primary.main',
                            filter: 'drop-shadow(0px 2px 4px rgba(0, 0, 0, 0.1))'
                          }} />
                        </motion.div>
                        <Typography 
                          variant="h6" 
                          color="text.primary"
                          sx={{
                            fontSize: '1.1rem',
                            fontWeight: 500
                          }}
                        >
                          {isDragActive
                            ? 'Drop the file here'
                            : 'Drag and drop your file here'}
                        </Typography>
                      </Box>
                      <Box sx={{
                        borderTop: '1px solid',
                        borderColor: 'grey.200',
                        pt: 1.5,
                        width: '100%',
                        textAlign: 'center',
                        backgroundColor: 'rgba(255, 255, 255, 0.9)'
                      }}>
                        <Typography 
                          variant="body2" 
                          color="text.secondary"
                          sx={{ fontSize: '0.875rem' }}
                        >
                          JPEG, PNG only (max 10MB)
                        </Typography>
                      </Box>
                    </Box>
                  </motion.div>
                </div>
              </Box>

              {preview && (
                <Box sx={{
                  ...BOX_DIMENSIONS,
                  transition: 'all 0.3s ease'
                }}>
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <Paper 
                      elevation={2} 
                      sx={{ 
                        p: 2, 
                        backgroundColor: 'background.paper',
                        borderRadius: '12px',
                        overflow: 'hidden',
                        height: '100%',
                        display: 'flex',
                        flexDirection: 'column',
                        minHeight: '250px',
                        position: 'relative'
                      }}
                    >
                      <Box sx={{ 
                        flex: 1, 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'center',
                        position: 'relative',
                        mb: 2,
                        '&::after': {
                          content: '""',
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          right: 0,
                          bottom: 0,
                          boxShadow: 'inset 0 0 12px rgba(0,0,0,0.05)',
                          pointerEvents: 'none',
                          borderRadius: '8px'
                        }
                      }}>
                        <img
                          src={preview}
                          alt="Preview"
                          style={{ 
                            maxWidth: '100%',
                            maxHeight: '180px',
                            objectFit: 'contain',
                            borderRadius: '8px',
                          }}
                        />
                      </Box>
                      <Box sx={{ textAlign: 'center' }}>
                        <Button
                          variant="contained"
                          onClick={handleUpload}
                          disabled={loading}
                          sx={{
                            minWidth: '140px',
                            height: '36px',
                            fontSize: '0.95rem',
                            textTransform: 'none',
                            boxShadow: '0 2px 8px rgba(33, 150, 243, 0.2)',
                            '&:hover': {
                              boxShadow: '0 4px 12px rgba(33, 150, 243, 0.3)',
                            }
                          }}
                        >
                          {loading ? (
                            <CircularProgress size={20} color="inherit" />
                          ) : (
                            'Analyze Image'
                          )}
                        </Button>
                      </Box>
                    </Paper>
                  </motion.div>
                </Box>
              )}
            </Box>

            {error && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
                style={{
                  width: '100%',
                  maxWidth: '350px'
                }}
              >
                <Alert 
                  severity="error" 
                  sx={{ 
                    borderRadius: '8px',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                    fontSize: '0.875rem'
                  }}
                >
                  {error}
                </Alert>
              </motion.div>
            )}

            {result && (
              <motion.div
                style={{
                  width: '100%',
                  maxWidth: '550px'
                }}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <Paper
                  elevation={2}
                  sx={{
                    backgroundColor: theme.palette.primary.main,
                    color: 'white',
                    borderRadius: '12px',
                    position: 'relative',
                    overflow: 'hidden',
                    minHeight: '70px'
                  }}
                >
                  <Box sx={{
                    background: `linear-gradient(135deg, ${theme.palette.primary.dark} 0%, ${theme.palette.primary.main} 100%)`,
                    p: 1.5,
                  }}>
                    <Box sx={{ 
                      display: 'flex', 
                      flexDirection: 'column',
                      gap: 1,
                      position: 'relative',
                      zIndex: 1
                    }}>
                      <Box sx={{ 
                        display: 'flex', 
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        backgroundColor: 'rgba(255, 255, 255, 0.1)',
                        borderRadius: '8px',
                        p: 1.5,
                        gap: 3
                      }}>
                        <Box sx={{ 
                          display: 'flex', 
                          alignItems: 'center', 
                          gap: 1.5,
                          flex: 1
                        }}>
                          <motion.div
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            transition={{ 
                              type: "spring",
                              stiffness: 260,
                              damping: 20
                            }}
                          >
                            <CheckCircle sx={{ 
                              fontSize: 24,
                              color: '#4caf50'
                            }} />
                          </motion.div>
                          <Box sx={{ flex: 1 }}>
                            <Typography sx={{ 
                              fontSize: '0.8rem',
                              opacity: 0.9,
                              mb: 0.25,
                              letterSpacing: '0.5px'
                            }}>
                              Classification Result
                            </Typography>
                            <Typography sx={{ 
                              fontSize: '1.1rem',
                              fontWeight: 600,
                              lineHeight: 1.1
                            }}>
                              {result.class}
                            </Typography>
                          </Box>
                        </Box>
                        <Box sx={{
                          backgroundColor: 'rgba(255, 255, 255, 0.15)',
                          borderRadius: '8px',
                          p: 1.25,
                          minWidth: '110px',
                          textAlign: 'center'
                        }}>
                          <Typography sx={{ 
                            fontSize: '0.8rem',
                            opacity: 0.9,
                            mb: 0.25,
                            letterSpacing: '0.5px'
                          }}>
                            Confidence
                          </Typography>
                          <Typography sx={{ 
                            fontSize: '1.1rem',
                            fontWeight: 600,
                            lineHeight: 1.1
                          }}>
                            {(result.confidence * 100).toFixed(1)}%
                          </Typography>
                        </Box>
                      </Box>
                    </Box>
                  </Box>
                </Paper>
              </motion.div>
            )}
          </Box>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;
