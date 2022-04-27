import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './styles.module.css';

function Home() {
  const context = useDocusaurusContext();
  const {siteConfig = {}} = context;
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Description will go into a meta tag in <head />">
      <header className={clsx('hero hero--primary', styles.heroBanner)}>
        <div className="container">
          <h1 className="hero__title">{siteConfig.title}</h1>
          {/*<p className="hero__subtitle">{siteConfig.tagline}</p>*/}
          <div className={styles.buttons}>
            <Link
              className={clsx(
                'button button--outline button--secondary button--lg',
                styles.getStarted,
              )}
              to={useBaseUrl('https://github.com/qua-platform/qua-libs/')}>
              Get Started
            </Link>
          </div>
        </div>
      </header>
      <main>
          <section className={styles.features}>
            <div className="container">
              <div className="row">
                <p>
                    Hey! Glad to see you there!

                    Welcome to the QUA libraries repository. Your one-stop-shop for
                    a batteries-included QUA experience.
                    Note that this website is being revised, and in the meanwhile, we recommend that you view the
                    libraries through the <a href='https://github.com/qua-platform/qua-libs/'>Github repository</a>.
                </p>
              </div>
            </div>
          </section>
      </main>
    </Layout>
  );
}

export default Home;
