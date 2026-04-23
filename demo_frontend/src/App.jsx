import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import { MLProvider, useML } from './MLContext';
import Dashboard from './Dashboard';
import { ShoppingBag, ChevronRight, CheckCircle2 } from 'lucide-react';

// Wrapper to show "Preload Status" logic for each mock page
const PageWrapper = ({ title, children }) => {
  const { preloaded } = useML();
  const location = useLocation();
  const [loadStatus, setLoadStatus] = useState("Loading...");

  useEffect(() => {
    // Determine component name from path
    const path = location.pathname;
    let compName = 'Home';
    if (path !== '/') {
        const parts = path.split('/');
        const name = parts[parts.length - 1];
        compName = name.charAt(0).toUpperCase() + name.slice(1);
    }

    if (preloaded.has(compName)) {
        setLoadStatus("🚀 0ms Render (Preloaded via AI)");
    } else {
        setLoadStatus("🐌 250ms Render (Network Bound)");
    }
  }, [location.pathname, preloaded]);

  return (
    <div className="page-container fade-in">
        <div className={`status-pill ${loadStatus.includes('0ms') ? 'fast' : 'slow'}`}>
            {loadStatus}
        </div>
        <h1>{title}</h1>
        <div className="page-content">{children}</div>
    </div>
  )
};

// Mock Pages
const Home = () => (
  <PageWrapper title="Storefront">
     <p>Welcome to our tech store. Start browsing our catalog.</p>
     <Link to="/products" className="btn primary">View Products <ChevronRight size={16}/></Link>
  </PageWrapper>
);

const Products = () => (
    <PageWrapper title="Product Catalog">
       <div className="product-grid">
           {[1, 2, 3].map(i => (
               <div key={i} className="dummy-card">
                   <div className="img-placeholder"></div>
                   <div className="pc-text">Tech Item #{i}</div>
                   <Link to="/detail" className="btn secondary">Details</Link>
               </div>
           ))}
       </div>
    </PageWrapper>
);

const Detail = () => (
    <PageWrapper title="Product Detail">
        <div className="detail-layout">
           <div className="img-placeholder large"></div>
           <div className="d-info">
               <h2>Flagship Smartphone</h2>
               <p className="price">$999</p>
               <p>The ultimate tech gadget with neural processing features.</p>
               <div className="actions">
                    <Link to="/products" className="btn outline">Back</Link>
                    <Link to="/cart" className="btn primary">Add to Cart</Link>
               </div>
           </div>
        </div>
    </PageWrapper>
);

const Cart = () => (
    <PageWrapper title="Shopping Cart">
        <div className="cart-item">
            <span>Flagship Smartphone</span>
            <span>$999</span>
        </div>
        <div className="cart-total">
            Total: $999
        </div>
        <Link to="/checkout" className="btn primary block">Proceed to Checkout</Link>
    </PageWrapper>
);

const Checkout = () => (
    <PageWrapper title="Checkout Complete">
        <div className="success-state">
            <CheckCircle2 size={48} color="#4ade80" />
            <h2>Order Placed!</h2>
            <Link to="/" className="btn secondary">Return Home</Link>
        </div>
    </PageWrapper>
);

export default function App() {
  return (
    <BrowserRouter>
      <MLProvider>
        <div className="app-layout">
            <main className="content-area">
                <header className="top-nav">
                    <div className="logo"><ShoppingBag /> Adaptive-Load Store</div>
                    <nav>
                        <Link to="/">Home</Link>
                        <Link to="/products">Catalog</Link>
                        <Link to="/cart">Cart</Link>
                    </nav>
                </header>
                
                <div className="route-wrapper">
                    <Routes>
                        <Route path="/" element={<Home />} />
                        <Route path="/products" element={<Products />} />
                        <Route path="/detail" element={<Detail />} />
                        <Route path="/cart" element={<Cart />} />
                        <Route path="/checkout" element={<Checkout />} />
                    </Routes>
                </div>
            </main>
            
            <aside className="dashboard-area">
                <Dashboard />
            </aside>
        </div>
      </MLProvider>
    </BrowserRouter>
  );
}
