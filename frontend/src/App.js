import React from "react";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import withReactSmart from "./reactsmart/withReactSmart";
import SmartRoute from "./reactsmart/SmartRoute";

function Home() {
  return <h1>Home Page</h1>;
}

function Products() {
  return <h1>Products Page</h1>;
}

const SmartHome = withReactSmart(Home);
const SmartProducts = withReactSmart(Products);

function App() {
  return (
    <BrowserRouter>
      <div style={{ padding: "20px" }}>
        <nav style={{ marginBottom: "20px" }}>
          <Link to="/" style={{ marginRight: "12px" }}>
            Home
          </Link>
          <Link to="/products">Products</Link>
        </nav>

        <SmartRoute>
          <Routes>
            <Route path="/" element={<SmartHome />} />
            <Route path="/products" element={<SmartProducts />} />
          </Routes>
        </SmartRoute>
      </div>
    </BrowserRouter>
  );
}

export default App;